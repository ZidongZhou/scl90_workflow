# Verification

- Package version: `1.0.0`. Python 3.11 is specified in `environment.yml`; `requirements.txt` pins the listed Python package versions.
- The full default pipeline was run with the separately downloaded public CSV. All 43 packaged analysis CSVs matched the regenerated tables, including the three renamed nested-CV outputs. All 22 supporting-information CSVs matched; the S1 workbook cell values matched as well.
- The five PNG previews were pixel-identical to the packaged figures. The SVG and PDF figure files matched byte-for-byte. The five submission TIFFs were pixel-identical, RGB, LZW-compressed, and saved at 600 dpi.
- The test suite passed 27 tests, including a quick pipeline run with the separately downloaded CSV, package structure, metadata, workbook numeric-cell types, and output-path safety. A quick run is not a substitute for the full analysis.
- This verification environment used Python 3.13.5 and XGBoost 3.1.3. `outputs/environment_report.json` records the runtime used for this verification; it is not a dependency lock file. The default workflow does not rerun the optional tree-based analysis, and S3 is an archived result. Tree-based outputs should be regenerated under the target environment before treating them as newly reproduced results.
- Arimo was used because genuine Arial was not installed in the verification environment. Font files are not included.
- The raw participant-level CSV is not included. Individual-level prediction files are not part of the release.
