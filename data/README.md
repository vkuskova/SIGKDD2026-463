# Data: `my_data.csv`

The democracy panel used in all experiments in the paper.

## Panel structure

- **139 countries**, each observed for **35 consecutive years (1990–2024)** → **4,865 rows** (139 × 35 = 4,865), a strongly balanced panel.
- **18 columns**: 2 identifier columns (`country_id`, `year`) + **16 democracy components**.
- Only countries with complete coverage across all 16 selected variables over the full window are retained; countries with incomplete coverage were excluded rather than imputed.
- All democracy variables are standardized to zero mean and unit variance.
- Sorted by `country_id`, then `year`. The pipeline relies on this ordering and on the equal segment length of 35 to construct lag windows that respect country boundaries (`split_timeseries = 35`).

## Columns and V-Dem variable mapping

The 16 democracy components are lower-level indices from V-Dem that serve as
building blocks for V-Dem's higher-order democracy indices. They are
expert-coded, bounded indicators aggregated to annual country–year observations.

| Column in `my_data.csv` | Component | V-Dem variable |
|---|---|---|
| `country_id` | Numeric country identifier (panel unit) | — |
| `year` | Calendar year (time index) | — |
| `Freedom_of_expression` | Freedom of expression and alternative information | `v2x_freexp_altinf` |
| `Freedom_of_association` | Freedom of association (thick) | `v2x_frassoc_thick` |
| `Suffrage` | Suffrage (share of population with voting rights) | `v2x_suffr` |
| `Clean_elections` | Clean elections | `v2xel_frefair` |
| `Elected_officials` | Elected officials | `v2x_elecoff` |
| `Individual_liberty` | Equality before the law and individual liberty | `v2xcl_rol` |
| `Judicial_constraints` | Judicial constraints on the executive | `v2x_jucon` |
| `Legislative_constraints` | Legislative constraints on the executive | `v2xlg_legcon` |
| `Civil_participation` | Civil society participation | `v2x_cspart` |
| `Direct_vote` | Direct popular vote | `v2xdd_dd` |
| `Local_government` | Local government elections | `v2xel_locelec` |
| `Regional_government` | Regional government elections | `v2xel_regelec` |
| `Deliberative` | Deliberative component | `v2xdl_delib` |
| `Equal_access` | Equal access | `v2xeg_eqaccess` |
| `Equal_distribution` | Equal distribution of resources | `v2xeg_eqdr` |
| `Equal_protection` | Equal protection | `v2xeg_eqprotec` |

## Source and license

Derived from the **Varieties of Democracy (V-Dem) v15** country-year dataset:

> Coppedge, Michael, et al. 2025. *V-Dem Country-Year Dataset v15.*
> Varieties of Democracy (V-Dem) Project.
> DOI: [10.23696/vdemds25](https://doi.org/10.23696/vdemds25)

V-Dem data is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
by the V-Dem Institute; this preprocessed subset is distributed under the same
terms. V-Dem project: <https://www.v-dem.net/>.
