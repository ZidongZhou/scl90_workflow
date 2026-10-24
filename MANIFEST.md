# File map

Main entry point: `code/run_all.py`. Main analyses are in `code/analysis.py`; figure generation is in `code/make_figures.py`.

- Submission figures: `figure_files_for_upload/Fig1.tif` through `Fig5.tif`.
- Preview/editable figure sources: `outputs/figures/` (PNG, SVG, PDF).
- Supporting information: `supporting_info/` (S1–S23 Tables).

## Archived inputs

The files in `code/static/` are aggregate historical inputs used by the default workflow. They are not recomputed unless the relevant option is explicitly enabled.

| File | Contents |
|---|---|
| `random_sets_1000.csv` | Archived random raw-score item-set distributions; 1,000 sets per item-set size |
| `random_summary_1000.csv` | Summary of the archived random-item distributions |
| `lasso_rank_hunan.csv` | Archived operational LASSO/L1 ranking from the mapped Hunan development group |
| `lasso_rank_shandong.csv` | Archived reverse-direction ranking from the mapped Shandong group |
| `lasso_stability_1000.csv` | Archived LASSO item-selection stability results |

## Supporting information

| File | Contents |
|---|---|
| S1_Table.xlsx | Pooled five-fold CV: summary and folds |
| S2_Table.csv | Archived LASSO stability |
| S3_Table.csv | Tree-based ranking sensitivity (archived) |
| S4_Table.csv | Random raw-score item-set distributions |
| S5_Table.csv | Current-environment LASSO sensitivity |
| S6_Table.csv | Data-quality exclusions |
| S7_Table.csv | Top-k bootstrap intervals |
| S8_Table.csv | Selective escalation |
| S9_Table.csv | Selective escalation bootstrap intervals |
| S10_Table.csv | Order-assumed domain coverage |
| S11_Table.csv | Nonoverlap and severity |
| S12_Table.csv | Nested item-selection CV |
| S13_Table.csv | Nested item stability |
| S14_Table.csv | Response-pattern audit |
| S15_Table.csv | Response-quality restricted workflow |
| S16_Table.csv | Nested threshold workflow |
| S17_Table.csv | Escalation error profile |
| S18_Table.csv | Subgroup diagnostics |
| S19_Table.csv | TRIPOD+AI reporting checklist |
| S20_Table.csv | Intended-use card |
| S21_Table.csv | Threshold trade-offs |
| S22_Table.csv | Retrospective quality audit |
| S23_Table.csv | First-stage observable quality flags |

