# navar_tvnar_pipeline.py
"""
NAVAR + tvNAR pipeline for democracy panel data.

This module:
1) Loads a panel CSV (expects country/time columns + variables),
2) Trains NAVAR to infer an influence/causal score matrix,
3) Converts NAVAR scores to a network W (threshold/top-k + row-normalize),
4) Fits tvNAR to get interpretable nodal influence trajectories lambda(tau),
5) Saves outputs (optional) and returns a structured results dict.

Designed to work with your existing NAVAR code:
- train_NAVAR from train_NAVAR.py
- split_timeseries support in dataloader.py (int length or list of lengths)

Dependencies: numpy, pandas, scikit-learn (optional), torch (via NAVAR)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from train_NAVAR import train_NAVAR
from tvNAR import fit_tvNAR_lambda_paths, fit_tvNAR_lambda_paths_p, row_normalize
from edge_ablation import edge_ablation_dm_all_pairs


@dataclass
class PanelSpec:
    country_col: str = "country_id"
    time_col: str = "year"
    variable_cols: Optional[List[str]] = None  # if None, infer as all non-id/time numeric cols
    sort: bool = True


def load_panel_csv(
    csv_path: Union[str, Path],
    panel: PanelSpec = PanelSpec(),
    dropna: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, List[int], List[str]]:
    """
    Load panel data and return:
        df_sorted: sorted dataframe
        Y: (T, N) stacked panel values
        lengths: list of per-country series lengths (sum to T)
        var_cols: variable column names used in Y

    dropna=True drops any row with missing values in the variable set (safe default for NAVAR).
    """
    df = pd.read_csv(csv_path)
    if panel.country_col not in df.columns or panel.time_col not in df.columns:
        raise ValueError(f"CSV must contain columns {panel.country_col!r} and {panel.time_col!r}")

    # infer variable cols
    if panel.variable_cols is None:
        # choose numeric columns excluding ids
        cand = [c for c in df.columns if c not in (panel.country_col, panel.time_col)]
        # keep only numeric-like
        num = []
        for c in cand:
            if pd.api.types.is_numeric_dtype(df[c]):
                num.append(c)
        if not num:
            raise ValueError("Could not infer numeric variable columns. Provide PanelSpec.variable_cols.")
        var_cols = num
    else:
        var_cols = list(panel.variable_cols)

    if dropna:
        df = df.dropna(subset=var_cols).copy()

    if panel.sort:
        df = df.sort_values([panel.country_col, panel.time_col]).reset_index(drop=True)

    # compute per-country lengths
    lengths = df.groupby(panel.country_col, sort=False).size().tolist()

    Y = df[var_cols].to_numpy(dtype=float)
    return df, Y, lengths, var_cols


def navar_scores_to_network(
    causal_matrix: np.ndarray,
    method: str = "topk",
    top_k: int = 4,
    threshold: Optional[float] = None,
    quantile: Optional[float] = None,
    keep_self: bool = False,
    nonnegative: bool = True,
    normalize_rows: bool = True,
) -> np.ndarray:
    """
    Convert NAVAR causal score matrix into a tvNAR network W.

    causal_matrix: (N,N) where entry [i,j] is score i->j OR j->i depending on your convention.
    In your train_NAVAR, causal_matrix is created from contributions.view(N,N). Historically,
    in NAVAR literature it's often "from j to i" confusion.
    Here we adopt a clear convention for tvNAR:
        W[i, j] = weight from variable j (sender) to variable i (receiver).

    Therefore, you may need to transpose depending on how you interpret the NAVAR matrix.
    This function does NOT transpose automatically; the pipeline exposes a flag.

    method:
        - "topk": keep top_k incoming edges per receiver i (row-wise topk over columns)
        - "threshold": keep edges >= threshold
        - "quantile": keep edges >= np.quantile(scores, quantile)

    keep_self:
        tvNAR uses (W+I) anyway. Usually keep_self=False for W itself.

    nonnegative:
        NAVAR scores are typically nonnegative already (std of contributions), but keep for safety.

    normalize_rows:
        row-normalize W for tvNAR stability/interpretability.
    """
    S = np.asarray(causal_matrix, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(f"causal_matrix must be square, got {S.shape}")
    N = S.shape[0]

    W = S.copy()

    if nonnegative:
        W = np.maximum(W, 0.0)

    if not keep_self:
        np.fill_diagonal(W, 0.0)

    if method.lower() == "topk":
        if top_k <= 0:
            raise ValueError("top_k must be >= 1 for method='topk'")
        # keep top_k per row (incoming to i)
        mask = np.zeros_like(W, dtype=bool)
        for i in range(N):
            row = W[i]
            if np.all(row <= 0):
                continue
            # argsort descending
            idx = np.argsort(row)[::-1]
            idx = idx[: min(top_k, N)]
            mask[i, idx] = True
        W = np.where(mask, W, 0.0)

    elif method.lower() == "threshold":
        if threshold is None:
            raise ValueError("threshold must be provided for method='threshold'")
        W = np.where(W >= float(threshold), W, 0.0)

    elif method.lower() == "quantile":
        if quantile is None:
            raise ValueError("quantile must be provided for method='quantile'")
        q = float(quantile)
        if not (0.0 <= q <= 1.0):
            raise ValueError("quantile must be in [0,1]")
        thr = float(np.quantile(W[W > 0], q)) if np.any(W > 0) else 0.0
        W = np.where(W >= thr, W, 0.0)

    else:
        raise ValueError("method must be one of: 'topk', 'threshold', 'quantile'")

    if normalize_rows:
        W = row_normalize(W)

    return W

def _pvals_to_sig_mask(pvals_df: pd.DataFrame, alpha: float = 0.05) -> np.ndarray:
    """
    pvals_df: rows=source, cols=target (as returned by edge_ablation_dm_all_pairs)
    returns: boolean mask (N,N) where True indicates significant source->target.
    """
    P = pvals_df.to_numpy(dtype=float)
    sig = (P < alpha)
    np.fill_diagonal(sig, False)
    return sig

def build_W_from_scores_with_options(
    *,
    causal_matrix: np.ndarray,
    var_cols: List[str],
    model,
    Y: np.ndarray,
    maxlags: int,
    split_spec,
    navar_matrix_transpose: bool = False,

    # Base network method (topk/threshold/quantile)
    network_method: str = "topk",          # "topk" | "threshold" | "quantile"
    top_k: int = 4,
    threshold: Optional[float] = None,
    quantile: Optional[float] = None,
    keep_self: bool = False,
    normalize_rows: bool = True,

    # Edge ablation options
    use_edge_ablation: bool = False,
    edge_ablation_keep_all_sig: bool = True,   # True => keep ALL significant edges, ignore top-k/threshold/quantile
    alpha: float = 0.05,
    ablation_val_proportion: float = 0.10,
    ablation_normalize_for_eval: bool = True,
    ablation_batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """
    Build a tvNAR network W from NAVAR scores with optional edge-ablation significance filtering.

    Returns:
      W: (N,N) receiver(rows) x sender(cols) network (tvNAR convention W[receiver, sender])
      S_used: the oriented NAVAR score matrix used to build W (same orientation as W)
      artifacts: dict with ablation outputs when run

    Conventions:
      - W[i, j] = sender j -> receiver i (rows receive, cols send)
      - edge_ablation_dm_all_pairs returns pvals_df with rows=source (sender), cols=target (receiver)
        so we transpose that significance mask into receiver x sender.
    """
    # 1) Orient NAVAR scores into receiver x sender orientation used by navar_scores_to_network
    S = causal_matrix.T if navar_matrix_transpose else causal_matrix
    S = np.asarray(S, dtype=float)

    artifacts: Dict[str, object] = {
        "network_method": network_method,
        "use_edge_ablation": bool(use_edge_ablation),
        "edge_ablation_keep_all_sig": bool(edge_ablation_keep_all_sig),
        "alpha": float(alpha),
        "pvals_df": None,
        "dm_df": None,
        "diff_df": None,
        "sig_mask_source_target": None,     # sender x receiver
        "sig_mask_receiver_sender": None,   # receiver x sender
    }

    # 2) Optional: edge ablation significance mask
    sig_mask_rs = None  # receiver x sender
    if use_edge_ablation:
        split_for_ablation = split_spec if isinstance(split_spec, int) else None

        pvals_df, dm_df, diff_df = edge_ablation_dm_all_pairs(
            model=model,
            data=Y,
            variable_names=var_cols,
            maxlags=maxlags,
            split_timeseries=split_for_ablation,
            val_proportion=ablation_val_proportion,
            normalize_for_eval=ablation_normalize_for_eval,
            batch_size=ablation_batch_size,
            include_diagonal=False,
            verbose=True,
        )

        sig_st = _pvals_to_sig_mask(pvals_df, alpha=alpha)  # sender x receiver
        sig_mask_rs = sig_st.T                              # receiver x sender

        artifacts["pvals_df"] = pvals_df
        artifacts["dm_df"] = dm_df
        artifacts["diff_df"] = diff_df
        artifacts["sig_mask_source_target"] = sig_st
        artifacts["sig_mask_receiver_sender"] = sig_mask_rs

        # If keep-all-significant mode: zero out all non-significant edges in S before building W
        if edge_ablation_keep_all_sig:
            S = np.where(sig_mask_rs, S, 0.0)

    # 3) Base method selection
    method = network_method.lower().strip()
    if method not in {"topk", "threshold", "quantile"}:
        raise ValueError("network_method must be one of: 'topk', 'threshold', 'quantile'")

    # If keep-all-significant mode is ON, we ignore topk/threshold/quantile selection and just
    # row-normalize the significant-edge-weighted matrix.
    if edge_ablation_keep_all_sig and use_edge_ablation:
        W = navar_scores_to_network(
            causal_matrix=S,
            method="threshold",
            threshold=0.0,      # keep all nonzero (already significance-masked)
            top_k=top_k,        # unused for threshold
            quantile=None,
            keep_self=keep_self,
            nonnegative=True,
            normalize_rows=normalize_rows,
        )
        return W, S, artifacts

    # Otherwise: build W from S using the requested base method
    W = navar_scores_to_network(
        causal_matrix=S,
        method=method,
        top_k=top_k,
        threshold=threshold,
        quantile=quantile,
        keep_self=keep_self,
        nonnegative=True,
        normalize_rows=normalize_rows,
    )

    # 4) If ablation is enabled and we are NOT in keep-all mode, intersect with significance
    if use_edge_ablation and (sig_mask_rs is not None):
        W = np.where(sig_mask_rs, W, 0.0)
        if normalize_rows:
            W = row_normalize(W)

    return W, S, artifacts

def run_navar_tvnar(
    csv_path: Union[str, Path],
    out_dir: Optional[Union[str, Path]] = None,
    panel: PanelSpec = PanelSpec(),
    dropna: bool = True,
    # NAVAR params
    maxlags: int = 8,
    tvnar_lags: int = 1,
    hidden_nodes: int = 128,
    dropout: float = 0.0,
    epochs: int = 500,
    learning_rate: float = 3e-4,
    batch_size: int = 256,
    lambda1: float = 0.0,
    val_proportion: float = 0.0,
    weight_decay: float = 0.0,
    hidden_layers: int = 1,
    normalize: bool = True,
    lstm: bool = False,
    # network conversion params
    network_method: str = "topk",   # "topk" | "ablation" | "both" | "none"
    navar_matrix_transpose: bool = False,
    top_k: int = 4,
    keep_self: bool = False,
    threshold: Optional[float] = None,
    quantile: Optional[float] = None,
    normalize_rows: bool = True,
    # edge ablation params (NEW)
    use_edge_ablation: bool = False,
    edge_ablation_keep_all_sig: bool = True,
    alpha: float = 0.05,
    ablation_val_proportion: float = 0.10,
    ablation_normalize_for_eval: bool = True,
    ablation_batch_size: int = 512,
    # tvNAR params
    grid_size: int = 50,
    bandwidth: float = 0.2,
    kernel: str = "gaussian",
    ridge: float = 1e-4,
    tau_mode: str = "within_series",
    center: bool = False,
) -> Dict[str, object]:
    """
    End-to-end NAVAR -> W -> tvNAR pipeline.

    Returns dict with:
      - df_sorted
      - var_cols
      - lengths
      - navar: causal_matrix, val_loss
      - W
      - tvnar: tau_grid, lambda, meta
      - saved_paths (if out_dir provided)
    """
    df, Y, lengths, var_cols = load_panel_csv(csv_path, panel=panel, dropna=dropna)

    # split_timeseries can be list[int] (unbalanced) or int (balanced).
    split_spec: Union[int, List[int]]
    # If all lengths identical, pass int to match your prior code; else list.
    if len(set(lengths)) == 1:
        split_spec = lengths[0]
    else:
        split_spec = lengths

    # Train NAVAR
    causal_matrix, contributions, val_loss, model = train_NAVAR(
        data=Y,
        maxlags=maxlags,
        hidden_nodes=hidden_nodes,
        dropout=dropout,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        lambda1=lambda1,
        val_proportion=val_proportion,
        weight_decay=weight_decay,
        check_every=max(1, epochs // 10),
        hidden_layers=hidden_layers,
        normalize=normalize,
        split_timeseries=split_spec,
        lstm=lstm,
    )

    S = causal_matrix.T if navar_matrix_transpose else causal_matrix

    # Convert NAVAR scores -> W
    W, S_used, artifacts = build_W_from_scores_with_options(
        causal_matrix=causal_matrix,
        var_cols=var_cols,
        model=model,
        Y=Y,
        maxlags=maxlags,
        split_spec=split_spec,
        navar_matrix_transpose=navar_matrix_transpose,
       network_method=network_method,
        top_k=top_k,
        threshold=threshold,
        quantile=quantile,
        keep_self=keep_self,
        normalize_rows=normalize_rows,
        use_edge_ablation=use_edge_ablation,
        edge_ablation_keep_all_sig=edge_ablation_keep_all_sig,
        alpha=alpha,
        ablation_val_proportion=ablation_val_proportion,
        ablation_normalize_for_eval=ablation_normalize_for_eval,
        ablation_batch_size=ablation_batch_size,
    )

    # Fit tvNAR
    if tvnar_lags <= 1:
      tv = fit_tvNAR_lambda_paths(
        Y=Y,
        W=W,
        split_timeseries=split_spec,
        grid_size=grid_size,
        bandwidth=bandwidth,
        kernel=kernel,
        ridge=ridge,
        tau_mode=tau_mode,
        center=center,
      )
    else:
      tv = fit_tvNAR_lambda_paths_p(
        Y=Y,
        W=W,
        split_timeseries=split_spec,
        grid_size=grid_size,
        bandwidth=bandwidth,
        kernel=kernel,
        ridge=ridge,
        tau_mode=tau_mode,
        center=center,
        tvnar_lags=tvnar_lags,
      )


    results: Dict[str, object] = {
        "df_sorted": df,
        "var_cols": var_cols,
        "lengths": lengths,
        "navar": {
            "causal_matrix": causal_matrix,
            "val_loss": val_loss,
            "model": model,                 # ← ADD THIS
            "contributions": contributions, # ← ADD THIS
            "network_artifacts": artifacts,   # <-- NEW
        },
        "W": W,
        "tvnar": {
            "tau_grid": tv["tau_grid"],
            "lambda": tv["lambda"],
            "W": W,
            "meta": tv["meta"],
        },
    }

    # Optional saves
    saved = {}
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save NAVAR causal matrix
        causal_df = pd.DataFrame(causal_matrix, index=var_cols, columns=var_cols)
        p_causal = out_dir / "navar_causal_matrix.csv"
        causal_df.to_csv(p_causal)
        saved["navar_causal_matrix"] = str(p_causal)

        # Save W
        W_df = pd.DataFrame(W, index=var_cols, columns=var_cols)
        p_W = out_dir / "W_navar_tvnar.csv"
        W_df.to_csv(p_W)
        saved["W"] = str(p_W)

        # Save lambda paths
        lam = tv["lambda"]
        tau_grid = tv["tau_grid"]
        lam_df = pd.DataFrame(lam, columns=[f"lambda_{c}" for c in var_cols])
        lam_df.insert(0, "tau", tau_grid)
        p_lam = out_dir / "tvnar_lambda_paths.csv"
        lam_df.to_csv(p_lam, index=False)
        saved["lambda_paths"] = str(p_lam)

        # Save edge ablation artifacts (if computed)
        na = results["navar"].get("network_artifacts", {})
        if na.get("use_edge_ablation") and na.get("pvals_df") is not None:
          na["pvals_df"].to_csv(out_dir / "edge_ablation_dm_pvalues.csv")
          na["dm_df"].to_csv(out_dir / "edge_ablation_dm_stats.csv")
          na["diff_df"].to_csv(out_dir / "edge_ablation_dm_mean_diff.csv")

        # Save meta / config
        meta = {
            "panel": panel.__dict__,
            "split_spec": split_spec if isinstance(split_spec, int) else list(split_spec),
            "navar_params": {
                "maxlags": maxlags,
                "hidden_nodes": hidden_nodes,
                "dropout": dropout,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "lambda1": lambda1,
                "val_proportion": val_proportion,
                "weight_decay": weight_decay,
                "hidden_layers": hidden_layers,
                "normalize": normalize,
                "lstm": lstm,
            },
            "network_params": {
                "navar_matrix_transpose": navar_matrix_transpose,
                "network_method": network_method,
                "top_k": top_k,
                "threshold": threshold,
                "quantile": quantile,
                "keep_self": keep_self,
                "normalize_rows": normalize_rows,
                # NEW — edge ablation controls
                "use_edge_ablation": use_edge_ablation,
                "edge_ablation_keep_all_sig": edge_ablation_keep_all_sig,
                "alpha": alpha,
                "ablation_val_proportion": ablation_val_proportion,
                "ablation_normalize_for_eval": ablation_normalize_for_eval,
                "ablation_batch_size": ablation_batch_size,
            },
            "tvnar_params": {
                "grid_size": grid_size,
                "bandwidth": bandwidth,
                "kernel": kernel,
                "ridge": ridge,
                "tau_mode": tau_mode,
                "center": center,
                "tvnar_lags": tvnar_lags, 
            },
            "tvnar_meta": tv["meta"],
        }
        p_meta = out_dir / "pipeline_meta.json"
        p_meta.write_text(json.dumps(meta, indent=2))
        saved["meta"] = str(p_meta)

    results["saved_paths"] = saved
    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run NAVAR -> tvNAR pipeline on democracy panel data.")
    p.add_argument("--csv", required=True, help="Path to panel CSV (country_id, year, variables...).")
    p.add_argument("--out", default=None, help="Output directory for CSV/JSON artifacts.")
    p.add_argument("--country_col", default="country_id")
    p.add_argument("--time_col", default="year")
    p.add_argument("--vars", default=None, help="Comma-separated variable columns (optional).")
    p.add_argument("--keepna", action="store_true", help="Do not drop rows with NA in variables.")

    # NAVAR
    p.add_argument("--maxlags", type=int, default=8)
    p.add_argument("--hidden_nodes", type=int, default=128)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lambda1", type=float, default=0.0)
    p.add_argument("--hidden_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--no_normalize", action="store_true")
    p.add_argument("--lstm", action="store_true")

    # Network conversion
    p.add_argument("--transpose_scores", action="store_true", help="Transpose NAVAR causal_matrix before building W.")
    p.add_argument("--network_method", default="topk", choices=["topk", "threshold", "quantile"])
    p.add_argument("--top_k", type=int, default=4)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--quantile", type=float, default=None)
    p.add_argument("--keep_self", action="store_true")
    p.add_argument("--no_row_norm", action="store_true")

    # tvNAR
    p.add_argument("--grid_size", type=int, default=50)
    p.add_argument("--bandwidth", type=float, default=0.2)
    p.add_argument("--kernel", default="gaussian", choices=["gaussian", "epanechnikov"])
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--tau_mode", default="within_series", choices=["within_series", "global_index"])
    p.add_argument("--center", action="store_true")

    return p


def main():
    args = _build_arg_parser().parse_args()

    var_cols = None
    if args.vars:
        var_cols = [v.strip() for v in args.vars.split(",") if v.strip()]

    panel = PanelSpec(
        country_col=args.country_col,
        time_col=args.time_col,
        variable_cols=var_cols,
        sort=True,
    )

    normalize = True
    if args.no_normalize:
        normalize = False
    elif args.normalize:
        normalize = True

    results = run_navar_tvnar(
        csv_path=args.csv,
        out_dir=args.out,
        panel=panel,
        dropna=(not args.keepna),
        maxlags=args.maxlags,
        hidden_nodes=args.hidden_nodes,
        dropout=args.dropout,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        lambda1=args.lambda1,
        hidden_layers=args.hidden_layers,
        normalize=normalize,
        lstm=args.lstm,
        navar_matrix_transpose=args.transpose_scores,
        network_method=args.network_method,
        top_k=args.top_k,
        threshold=args.threshold,
        quantile=args.quantile,
        keep_self=args.keep_self,
        normalize_rows=(not args.no_row_norm),
        grid_size=args.grid_size,
        bandwidth=args.bandwidth,
        kernel=args.kernel,
        ridge=args.ridge,
        tau_mode=args.tau_mode,
        center=args.center,
    )

    saved = results.get("saved_paths", {})
    if saved:
        print("Saved artifacts:")
        for k, v in saved.items():
            print(f"  {k}: {v}")
    else:
        print("Run complete (no outputs saved).")


if __name__ == "__main__":
    # In notebooks/Colab, argv may contain Jupyter/Colab flags.
    # Require explicit CLI invocation with --csv to run main().
    import sys

    if any(arg.startswith("--csv") for arg in sys.argv):
        main()
    else:
        print(
            "navar_tvnar_pipeline.py loaded as a module. "
            "To run from CLI, call: python navar_tvnar_pipeline.py --csv <path_to_csv> [--out <dir> ...]"
        )