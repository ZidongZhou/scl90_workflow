# SCL-90 selective-escalation workflow

Code and aggregate results for:

*A selective-escalation workflow for approximating the SCL-90 total-score screening rule among Chinese first-year college students: a retrospective simulation study.*

The workflow evaluates a 15-item first stage for routing records to full-form completion or priority human review. It approximates an existing total-score screening rule; it is not a diagnostic tool or a validated replacement for the full SCL-90.

## Data

Download `Data_collegestudent.csv` from [Mendeley Data](https://doi.org/10.17632/btzgmr2rt2.1). The raw dataset is not included. Place the file at `data_raw/Data_collegestudent.csv`, or pass its location with `--data`.

## Run

The recommended environment is Python 3.11, as specified in `environment.yml`. Install the listed dependencies and run from the project root:

```bash
python -m pip install -r requirements.txt
python code/run_all.py --data data_raw/Data_collegestudent.csv --out outputs_run
python -m pytest -q
```

Use a new output directory for each run. To replace an existing `outputs_*` directory, add `--clean`. The script refuses protected paths, symbolic links, and output paths containing the input data. A non-empty output directory cannot be reused without `--clean`.

The default run uses archived random-item baseline and LASSO stability results in `code/static/`. Add `--recompute-random-baseline` to recompute the random-item baseline, or `--include-tree` to run the optional LightGBM/XGBoost sensitivity analysis. The packaged S3 table is an archived result; the default run does not regenerate it.

A run with `--quick` is only a smoke test. It skips bootstrap resampling and nested cross-validation and should not be used as the full analysis run.

## Files

- `code/run_all.py` — main entry point.
- `code/analysis.py` — main analyses.
- `code/make_figures.py` — figures and summary output.
- `code/tree_sensitivity.py` — optional tree-based sensitivity analysis.
- `code/check_font.py` — figure-font check.
- `code/static/` — archived aggregate inputs used by the default workflow.
- `outputs/tables/` — packaged aggregate analysis tables.
- `outputs/figures/` — PNG previews and SVG/PDF figure files.
- `figure_files_for_upload/` — TIFF files for journal submission.
- `supporting_info/` — S1–S23 supporting tables.
- `MANIFEST.md` — supporting-table and archive-file map.

Figures use Arial when available. If it is not installed, the script uses a compatible sans-serif font where possible. Font files are not included.

## Citation

See `CITATION.cff` for the software citation. Cite the accompanying manuscript and the source dataset when using this work. The source dataset is distributed separately under the license listed on Mendeley Data.

Repository: https://github.com/ZidongZhou/scl90_workflow
