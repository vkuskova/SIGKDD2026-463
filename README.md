# DCNAR: From Causal Discovery to Dynamic Causal Inference in Neural Time Series

Reproducibility repository for the KDD '26 paper **"From Causal Discovery to
Dynamic Causal Inference in Neural Time Series"** (paper #463).

DCNAR is a two-stage neural causal framework. **Stage I** learns a sparse,
directed Granger-causal network from multivariate panel time series using
Neural Additive Vector Autoregression (NAVAR), refined into a structural prior
via forecast-necessity edge ablation. **Stage II** conditions a time-varying
network autoregression (tvNAR) on that learned prior, enabling impulse-response
and counterfactual analysis without a pre-specified causal structure. The
framework is evaluated on a 139-country × 35-year Varieties of Democracy
(V-Dem) panel.

- **Paper (ACM DOI):** <https://doi.org/10.1145/3770855.3818956>
- **Preprint (arXiv):** arXiv:2603.20980

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vkuskova/SIGKDD2026-463/blob/main/notebooks/KDD_463.ipynb)

The paper and its supplementary materials are included in this repository under
[`paper/`](paper/): the camera-ready main paper (`KDD463.pdf`) and the
supplementary materials (`KDD463_supplementary.pdf`), which contain the full
baseline specifications, evaluation-metric definitions, DCNAR implementation
detail, extended-panel analysis, and the bandwidth-sensitivity,
architectural-ablation, and coefficient-path results referenced below.

## Repository structure

```
SIGKDD2026-463/
├── README.md                      This file
├── LICENSE                        CC-BY-4.0
├── requirements.txt               Python dependencies
├── paper/
│   ├── KDD463.pdf                 Main paper (camera-ready)
│   └── KDD463_supplementary.pdf   Supplementary materials
├── notebooks/
│   └── KDD_463.ipynb              End-to-end Colab orchestrator (run this)
├── src/                           Pipeline modules (imported by the notebook)
│   ├── NAVAR.py                   NAVAR model (MLP/Conv and LSTM variants)
│   ├── train_NAVAR.py             NAVAR training loop -> causal score matrix S
│   ├── dataloader.py              Lag-window construction (panel-aware)
│   ├── edge_ablation.py           Diebold–Mariano forecast-necessity edge tests
│   ├── tvNAR.py                   Time-varying network autoregression (Stage II)
│   ├── navar_tvnar_pipeline.py    Orchestration: scores -> structural prior -> tvNAR
│   ├── evaluate.py                Scoring utilities (e.g. AUROC)
│   └── run_NAVAR.py               Standalone CLI for NAVAR only
├── data/
│   ├── my_data.csv                The 139 x 35 democracy panel (16 components)
│   └── README.md                  Data dictionary + V-Dem provenance
└── results/                       Outputs reproduced by the notebook
    ├── bandwidth_sensitivity/     Bandwidth sweep (supp. Sec. 5)
    ├── architectural_ablation/    Component ablation (main Table 1 / supp. Sec. 6)
    └── lambda_trajectories/       Coefficient paths + effective networks (supp. Sec. 7)
```

## How to reproduce

The pipeline is designed to run end-to-end on **Google Colab** with a GPU.

1. Open `notebooks/KDD_463.ipynb` in Colab
   (Upload, or File → Open notebook → GitHub → this repository).
2. Enable a GPU: **Runtime → Change runtime type → GPU**.
3. **Run the first setup cell.** It clones this repository into the Colab
   session, puts `src/` on the Python path, and (optionally) mounts Google
   Drive to persist outputs. No manual file uploads are required.
4. **Run the remaining cells top to bottom.** Each step is a self-contained
   section; the notebook trains NAVAR, builds the structural prior, fits
   tvNAR and the baselines, and regenerates the figures and tables.

### Where inputs and outputs live

- **Inputs** (the `src/` modules and `data/my_data.csv`) come from this
  repository, pulled by the clone step in the first cell. The notebook's
  imports (`from train_NAVAR import ...`) and data path
  (`SIGKDD2026-463/data/my_data.csv`) are configured for this layout.
- **Outputs** are written under `outputs/` in the session and, if Drive is
  mounted, copied to `MyDrive/DCNAR/KDD_463_Responses`. Colab sessions are
  ephemeral, so use Drive (or download manually) to keep results. The
  canonical outputs are already committed under `results/` for reference.

### Running locally (without Colab)

The modules in `src/` are plain Python and run anywhere PyTorch is available.
Install dependencies and either work through the notebook, or call the NAVAR
stage directly:

```bash
pip install -r requirements.txt
python src/run_NAVAR.py --filename data/my_data.csv --maxlags 8
```

A CUDA-capable GPU is recommended but not required; NAVAR training completes in
roughly 2–3 minutes on a single GPU for this panel.

## Key configuration

The notebook's `CFG` dataclass holds all settings used in the paper. The most
important:

| Setting | Value | Meaning |
|---|---|---|
| `maxlags` | 8 | Maximum lag length for NAVAR causal discovery. |
| `tvnar_lags` | 1 | tvNAR(1) is the primary specification in the paper. |
| `train_years` / `val_years` | 25 / 10 | Per-country chronological split. |
| `H_eval` / `H_irf` | 10 / 10 | Forecast and impulse-response horizons. |
| `top_k` | 4 | Edges retained per target when building the prior. |
| `alpha_dm` | 0.05 | Significance level for Diebold–Mariano edge tests. |

## Results map

The committed outputs correspond to the paper as follows:

- **`results/bandwidth_sensitivity/`** — the six-value bandwidth sweep
  (h ∈ {0.05, …, 0.40}); supports the claim that DCNAR's smooth impulse
  responses are not a smoothing artifact (supplementary Section 5).
- **`results/architectural_ablation/`** — leave-one-component-out ablation
  (neural discovery vs. VAR-derived structure, time-varying vs. static Λ,
  sparse vs. dense Ĝ); main paper Table 1 and supplementary Section 6.
- **`results/lambda_trajectories/`** — time-varying node-influence paths
  λ(t), TV-VAR diagonal-coefficient comparison, and effective-network
  snapshots for Albania, the United States, and Mexico; supplementary
  Section 7. (Includes some diagnostic figures beyond those in the paper.)

## Code attribution

The core NAVAR architecture and training code — `src/NAVAR.py`,
`src/train_NAVAR.py`, `src/dataloader.py`, `src/evaluate.py`, and
`src/run_NAVAR.py` — was originally developed by:

> Bussmann, B., Nys, J., and Latré, S. 2021. Neural Additive Vector
> Autoregression Models for Causal Discovery in Time Series. In *Discovery
> Science*, Lecture Notes in Computer Science, vol. 12986. Springer, Cham.
> DOI: [10.1007/978-3-030-88942-5_27](https://doi.org/10.1007/978-3-030-88942-5_27)

Original code: <https://github.com/bartbussmann/NAVAR>.

The following modifications were made for this work:

- **`train_NAVAR.py`** — added GPU device handling; modified the return
  signature to also return the trained model object (4-value return); adapted
  loss computation for panel time series with country-segment boundaries;
  added a validation-loss recomputation pass.
- **`dataloader.py`** — extended `split_timeseries` to accept a per-unit length
  vector for unbalanced panels (the original assumed equal-length segments);
  added a guard that rejects variable-length `split_timeseries` when
  `lstm=True` (unsupported); fixed `np.int` → `int` in `split_train_val()` for
  NumPy ≥ 1.24 compatibility.
- **`run_NAVAR.py`** — updated the return unpacking from 3 to 4 values to match
  the modified `train_NAVAR` signature.

`NAVAR.py` and `evaluate.py` are used as released by Bussmann et al. (2021).

The remaining modules are new and developed for this research program:
`edge_ablation.py` (forecast-necessity testing via Diebold–Mariano edge
ablation; Kuskova et al., FLAIRS 2026) and `tvNAR.py` /
`navar_tvnar_pipeline.py` (the NAVAR → tvNAR dynamic causal inference pipeline
of this paper).

## Companion work

The forecast-necessity edge-ablation procedure used to construct the
structural prior is developed and benchmarked separately in:

> V. Kuskova, D. Zaytsev, and M. Coppedge. 2026. Beyond Coefficients:
> Forecast-Necessity Testing for Interpretable Causal Discovery in Nonlinear
> Time-Series Models. In *Proceedings of the 39th International Florida
> Artificial Intelligence Research Society Conference (FLAIRS)*.
> https://doi.org/10.32473/flairs.39.1.141791 (arXiv:2604.18751)

## Citation

```bibtex
@inproceedings{Zaytsev2026DCNAR,
  author    = {Zaytsev, Dmitry and Kuskova, Valentina V. and Coppedge, Michael},
  title     = {From Causal Discovery to Dynamic Causal Inference in Neural Time Series},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge
               Discovery and Data Mining V.2 (KDD '26)},
  year      = {2026},
  doi       = {10.1145/3770855.3818956}
}
```

## License

Except where noted below, this repository's contents (the new code, the data
subset, and the results) are released under the
[Creative Commons Attribution 4.0 International License](LICENSE) (CC BY 4.0).

Two exceptions:

- The NAVAR-derived modules (`src/NAVAR.py`, `src/train_NAVAR.py`,
  `src/dataloader.py`, `src/evaluate.py`, `src/run_NAVAR.py`) are adapted from
  Bussmann et al. (2021) and remain subject to their original license terms;
  see the Code Attribution section above.
- `data/my_data.csv` is derived from the Varieties of Democracy (V-Dem) v15
  dataset, released by the V-Dem Institute under CC BY 4.0; see
  `data/README.md` for provenance and citation.
