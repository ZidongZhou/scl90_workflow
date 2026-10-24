# Figure fonts

The package does not redistribute Arial font files. If you have a properly licensed Arial installation, make it available to Matplotlib before regenerating figures. On Linux, system-installed `.ttf` files can be refreshed with `fc-cache -f`; then run `python code/check_font.py`.

The figure script selects genuine Arial when available. Otherwise it uses Arimo, a metrically compatible substitute, or another sans-serif fallback if Arimo is unavailable. The selected font family and font file name are recorded in `outputs/figure_font_report.json`.
