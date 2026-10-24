#!/usr/bin/env python3
"""Create data-derived manuscript Figures 1-5 and a JSON summary from generated output tables."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

def _resolve_figure_font() -> tuple[str, str, bool]:
    """Use genuine Arial when installed; otherwise use an Arial-metric-compatible fallback."""
    for candidate in ("Arial", "Arimo", "Liberation Sans", "Nimbus Sans"):
        try:
            font_path = font_manager.findfont(
                font_manager.FontProperties(family=candidate),
                fallback_to_default=False,
            )
            actual_family = font_manager.FontProperties(fname=font_path).get_name()
        except (ValueError, OSError):
            continue
        if actual_family.casefold() == candidate.casefold():
            return actual_family, font_path, actual_family.casefold() == "arial"
    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"))
    return font_manager.FontProperties(fname=path).get_name(), path, False


FIGURE_FONT_FAMILY, FIGURE_FONT_PATH, TRUE_ARIAL_AVAILABLE = _resolve_figure_font()

# Arial is preferred. Proprietary Arial font files are not redistributed with
# this package; Arimo is used when genuine Arial is not installed locally.
plt.rcParams.update({
    "font.family": FIGURE_FONT_FAMILY,
    "font.sans-serif": [FIGURE_FONT_FAMILY],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "svg.fonttype": "none",
    "svg.hashsalt": "scl90-selective-escalation",
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

FIGURE_SPECS = {
    "figure1_burden_performance_frontier": "Fig1",
    "figure2_random_itemset_distribution": "Fig2",
    "figure3_selective_escalation_workflow": "Fig3",
    "figure4_capacity_coverage_curve": "Fig4",
    "figure5_calibration_recalibration": "Fig5",
}


def _read_first_existing(*paths: Path) -> pd.DataFrame:
    for path in paths:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError("None of the expected input tables exists: " + ", ".join(str(p) for p in paths))


def _save_figure(fig, figures_dir: Path, upload_dir: Path, stem: str) -> tuple[str, ...]:
    """Save one upload TIFF and separate preview/editable source files."""
    upload_stem = FIGURE_SPECS[stem]
    png_path = figures_dir / f"{stem}.png"
    svg_path = figures_dir / f"{stem}.svg"
    pdf_path = figures_dir / f"{stem}.pdf"
    tif_path = upload_dir / f"{upload_stem}.tif"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight", metadata={"Date": None})
    fig.savefig(pdf_path, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(tif_path, dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    with Image.open(tif_path) as tif:
        dpi = tif.info.get("dpi", (600, 600))
        rgb = tif.convert("RGB")
        rgb.save(tif_path, compression="tiff_lzw", dpi=dpi)
    return (
        str(png_path.name), str(svg_path.name), str(pdf_path.name),
        str(tif_path.relative_to(upload_dir.parent)),
    )


def _panel_label(ax, label: str) -> None:
    ax.text(
        -0.13,
        1.08,
        label,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=8,
        fontweight="bold",
        clip_on=False,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.8),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=Path("outputs_run"), type=Path)
    ap.add_argument("--upload-dir", default=None, type=Path, help="Directory for the five journal-upload TIFFs. Defaults to <out>/figure_files_for_upload.")
    args = ap.parse_args()
    T = args.out / "tables"
    F = args.out / "figures"
    F.mkdir(parents=True, exist_ok=True)
    upload_dir = args.upload_dir if args.upload_dir is not None else args.out / "figure_files_for_upload"
    upload_dir.mkdir(parents=True, exist_ok=True)
    # Remove stale duplicate exports left by older package versions.
    for stem, upload_stem in FIGURE_SPECS.items():
        for suffix in (".tif",):
            stale = F / f"{stem}{suffix}"
            if stale.exists():
                stale.unlink()
        for suffix in (".png", ".svg", ".pdf"):
            stale = upload_dir / f"{upload_stem}{suffix}"
            if stale.exists():
                stale.unlink()

    sample = pd.read_csv(T / "table1_sample_characteristics_quality.csv")
    perf = pd.read_csv(T / "table3_performance_by_item_count_and_selection.csv")
    random_dist = _read_first_existing(
        T / "random_rawsum_item_set_distributions.csv",
    )
    random_sum = _read_first_existing(
        T / "table5_random_rawsum_item_set_summary.csv",
    )
    triage = pd.read_csv(T / "table10_triage_capacity_coverage.csv")
    selective = _read_first_existing(T / "table11_selective_full_form_escalation.csv")
    threshold_tradeoff = pd.read_csv(T / "tableS_selective_escalation_threshold_tradeoff.csv")
    calib = pd.read_csv(T / "calibration_curve_bins_figure5.csv")
    lasso = perf[(perf.method == "LASSO/L1 ranking") & (perf.item_count.isin([5, 10, 15, 20, 30]))].sort_values("item_count")

    generated_figures = []
    journal_upload_figures = []

    # Fig 1: separate ranking/discrimination metrics and probability error.
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.45), dpi=300)
    palette = {"ROC-AUC": "#286d92", "PR-AUC": "#d6812d", "Top-10 recall": "#36846f"}
    for col, lab, style, marker in [
        ("roc_auc", "ROC-AUC", "-", "o"),
        ("pr_auc", "PR-AUC", "--", "s"),
        ("top10_recall", "Top-10 recall", "-.", "D"),
    ]:
        axes[0].plot(lasso.item_count, lasso[col], marker=marker, lw=1.5,
                     color=palette[lab], linestyle=style, label=lab, markersize=4.5)
    ceiling10 = float(triage.loc[triage.capacity_percent.eq(10), "theoretical_max_recall_given_capacity"].iloc[0])
    axes[0].axhline(ceiling10, color="#919aa2", lw=0.9, linestyle=":")
    axes[0].set(xlabel="Number of items", ylabel="Discrimination / top-10 recall")
    axes[0].set_xticks([5, 10, 15, 20, 30])
    axes[0].set_ylim(.42, 1.035)
    axes[0].legend(loc="center right", frameon=False, labelspacing=0.7)
    axes[1].plot(lasso.item_count, lasso["brier"], marker="o", lw=1.6,
                 markersize=4.5, color="#286d92")
    axes[1].set(xlabel="Number of items", ylabel="Brier score (lower is better)")
    axes[1].set_xticks([5, 10, 15, 20, 30])
    for panel, ax in zip("ab", axes):
        _panel_label(ax, panel)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.89, bottom=0.20, wspace=0.28)
    paths = _save_figure(fig, F, upload_dir, "figure1_burden_performance_frontier")
    generated_figures.extend(paths[:3]); journal_upload_figures.extend(paths[3:]); plt.close(fig)

    # Fig 2: explicitly distinguish raw-score distributions from fitted models.
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.45), dpi=300)
    for panel, ax, metric, lab in [
        ("a", axes[0], "roc_auc", "ROC-AUC"), ("b", axes[1], "pr_auc", "PR-AUC")
    ]:
        vals = [random_dist.loc[random_dist.item_count.eq(k), metric].to_numpy() for k in [5, 10, 15, 20, 30]]
        bp = ax.boxplot(vals, positions=[5, 10, 15, 20, 30], widths=1.9,
                        patch_artist=True, showfliers=False, manage_ticks=False)
        for box in bp["boxes"]:
            box.set(facecolor="#dce8ec", edgecolor="#456b7f", linewidth=0.85)
        for group in ("whiskers", "caps"):
            for element in bp[group]: element.set(color="#456b7f", linewidth=0.85)
        for element in bp["medians"]: element.set(color="#b56f2b", linewidth=1.25)
        ax.plot(lasso.item_count, lasso[metric], marker="o", color="#b45535",
                linestyle="--", linewidth=1.35, markersize=4.2, label="Fixed logistic reference")
        ax.set(xlabel="Number of items", ylabel=lab)
        ax.set_xticks([5, 10, 15, 20, 30])
        ax.legend(loc="lower right", frameon=False)
        _panel_label(ax, panel)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.88, bottom=0.19, wspace=0.26)
    paths = _save_figure(fig, F, upload_dir, "figure2_random_itemset_distribution")
    generated_figures.extend(paths[:3]); journal_upload_figures.extend(paths[3:]); plt.close(fig)

    # Fig 3: 2 + 1 panel layout preserves readable >= 8-pt text.
    holdout = selective[selective["sample"].str.contains("Shandong", case=False, na=False)].iloc[0]
    fig = plt.figure(figsize=(7.35, 5.25), dpi=300)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.95, 1.14], hspace=0.63,
                           wspace=0.42, left=0.13, right=0.88, top=0.93, bottom=0.12)
    ax_route = fig.add_subplot(grid[0, 0]); ax_burden = fig.add_subplot(grid[0, 1]); ax_trade = fig.add_subplot(grid[1, :])
    values = [int(holdout[c]) for c in ("low_zone_n", "middle_escalation_n", "high_zone_n")]
    cats = ["Provisional low", "Full-form", "Priority review"]
    cols = ["#7fb9cd", "#e6bd68", "#d18b5a"]
    y = list(range(3))
    ax_route.barh(y, [v / sum(values) * 100 for v in values], color=cols, height=0.58)
    ax_route.set_yticks(y, cats); ax_route.invert_yaxis()
    ax_route.set_xlim(0, 93); ax_route.set_xlabel("Share of hold-out group (%)")
    ax_route.set_title("a  Routing distribution", loc="left", fontweight="bold", pad=13)
    for j, v in enumerate(values):
        ax_route.text(v / sum(values) * 100 + 2, j, f"{v:,} ({100*v/sum(values):.1f}%)", va="center", ha="left", fontsize=8)
    ax_route.spines["left"].set_visible(False)
    ax_route.tick_params(axis="y", length=0)
    ax_route.set_xticks([0, 25, 50, 75])

    mean_items = float(holdout["average_item_burden"])
    ax_burden.barh([0, 1], [mean_items, 90], color=["#518da9", "#a5b0b6"], height=0.51)
    ax_burden.set_yticks([0, 1], ["Selective", "Full 90 items"])
    ax_burden.invert_yaxis()
    ax_burden.set_xlim(0, 113); ax_burden.set_xticks([0, 30, 60, 90])
    ax_burden.set_xlabel("Mean items per student")
    ax_burden.set_title("b  Questionnaire burden", loc="left", fontweight="bold", pad=13)
    ax_burden.tick_params(axis="y", length=0)
    ax_burden.spines["left"].set_visible(False)
    for j, v in enumerate([mean_items, 90]):
        ax_burden.text(v + 2, j, f"{v:.1f}", ha="left", va="center", fontsize=8)
    ax_burden.text(0.97, 0.97, f"{holdout['burden_reduction_vs_90']*100:.1f}% reduction",
                   transform=ax_burden.transAxes, ha="right", va="top", fontsize=8,
                   color="#4d5d67")

    low_trade = threshold_tradeoff[
        threshold_tradeoff["analysis"].eq("low-threshold trade-off with high threshold fixed")
    ].sort_values("low_cut_rawsum_le")
    xx = low_trade["low_cut_rawsum_le"].to_numpy()
    yy = low_trade["holdout_low_zone_npv"].to_numpy() * 100
    bb = low_trade["holdout_average_item_burden"].to_numpy()
    ax_trade.plot(xx, yy, marker="o", markersize=5, color="#286d92", linewidth=1.7, label="Low-zone NPV")
    ax_trade.axhline(99.5, color="#555e66", linestyle=":", lw=1.2, label="99.5% target")
    ax_trade.set_ylim(97.95, 100.14)
    ax_trade.set_ylabel("Low-zone NPV (%)", color="#286d92")
    ax_trade.tick_params(axis="y", labelcolor="#286d92")
    ax_trade.set_xticks([11,12,13,14], ["≤11", "≤12", "≤13", "≤14"])
    ax_trade.set_xlim(10.65, 14.35)
    ax_trade.set_xlabel("Provisional low-zone raw-sum threshold (high zone ≥24)")
    ax_trade.set_title("c  NPV versus questionnaire burden", loc="left", fontweight="bold", pad=12)
    ax2 = ax_trade.twinx()
    ax2.plot(xx, bb, marker="s", markersize=4.7, color="#bd7b30", linestyle="--", lw=1.5,
             label="Mean items")
    ax2.set_ylabel("Mean items per student", color="#a16b2b")
    ax2.tick_params(axis="y", labelcolor="#a16b2b")
    ax2.set_ylim(24.8, 33.0)
    ax2.spines["top"].set_visible(False)
    for at, color in [(12, "#397f94"), (13, "#bc7a2f")]:
        ax_trade.axvline(at, color=color, alpha=0.35, lw=0.9)
    lines1, labs1 = ax_trade.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax_trade.legend(lines1 + lines2, labs1 + labs2, loc="lower left", frameon=False, ncol=3,
                    fontsize=8, columnspacing=1.4, bbox_to_anchor=(0.01, 0.02))
    paths = _save_figure(fig, F, upload_dir, "figure3_selective_escalation_workflow")
    generated_figures.extend(paths[:3]); journal_upload_figures.extend(paths[3:]); plt.close(fig)

    # Fig 4: line patterns remain identifiable in grayscale.
    fig, ax = plt.subplots(figsize=(7.05, 4.30), dpi=300)
    ax.plot(triage.capacity_percent, triage.coverage_recall_of_full_score_screen_positive,
            marker="o", lw=1.65, markersize=5, color="#286d92", label="Observed recall")
    if "theoretical_max_recall_given_capacity" in triage.columns:
        ax.plot(triage.capacity_percent, triage.theoretical_max_recall_given_capacity,
                linestyle="--", lw=1.5, color="#c17b30", label="Theoretical recall ceiling")
    ax.plot(triage.capacity_percent, triage.positive_rate_in_reviewed_group,
            marker="s", linestyle="-.", lw=1.45, markersize=5, color="#397f69",
            label="Screen-positive among reviewed")
    ax.set(xlabel="Follow-up capacity (% of records)", ylabel="Proportion")
    ax.set_ylim(0, 1.05); ax.set_xticks([5, 10, 15, 20, 25, 30])
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.16, top=0.95)
    paths = _save_figure(fig, F, upload_dir, "figure4_capacity_coverage_curve")
    generated_figures.extend(paths[:3]); journal_upload_figures.extend(paths[3:]); plt.close(fig)

    # Fig 5: compare full range with a dedicated low-probability panel.
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.65), dpi=300)
    colormap = {"Original weighted model": "#bd7b30", "Slope+intercept recalibrated model": "#286d92"}
    for panel, ax in zip("ab", axes):
        ax.plot([0,1], [0,1], linestyle=":", lw=1, color="#666e73", label="Ideal calibration")
        for model, g in calib.groupby("model", sort=False):
            g = g.sort_values("mean_predicted_probability")
            ax.plot(g["mean_predicted_probability"], g["observed_screen_positive_rate"],
                    marker="o" if model == "Original weighted model" else "s",
                    linestyle="-" if model == "Original weighted model" else "--",
                    lw=1.6, markersize=4.4, color=colormap[model], label=model)
        ax.set(xlabel="Mean predicted probability", ylabel="Observed screen-positive rate")
        ax.set_aspect("equal", adjustable="box")
        if panel == "a":
            ax.set(xlim=(-0.02,1.02), ylim=(-0.02,1.02))
            ax.set_title("a  Full probability range", loc="left", fontweight="bold", pad=11)
        else:
            ax.set(xlim=(-0.008,0.27), ylim=(-0.008,0.30))
            ax.set_xticks([0, .1, .2]); ax.set_yticks([0, .1, .2, .3])
            ax.set_title("b  Low-probability detail", loc="left", fontweight="bold", pad=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.016))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.89, bottom=0.27, wspace=0.28)
    paths = _save_figure(fig, F, upload_dir, "figure5_calibration_recalibration")
    generated_figures.extend(paths[:3]); journal_upload_figures.extend(paths[3:]); plt.close(fig)

    font_report = {
        "preferred_font": "Arial",
        "font_family_used": FIGURE_FONT_FAMILY,
        "font_file_name": Path(FIGURE_FONT_PATH).name,
        "true_arial_available": TRUE_ARIAL_AVAILABLE,
        "note": ("Genuine Arial was used." if TRUE_ARIAL_AVAILABLE else
                 "Genuine Arial was not installed in the generation environment; Arimo or another named fallback was used. Arial font files are not bundled."),
    }
    with open(args.out / "figure_font_report.json", "w", encoding="utf-8") as f:
        json.dump(font_report, f, ensure_ascii=False, indent=2)
    if not TRUE_ARIAL_AVAILABLE:
        print(f"WARNING: Genuine Arial is not installed; figures use {FIGURE_FONT_FAMILY} ({FIGURE_FONT_PATH}).", flush=True)

    summary = {
        "sample": sample.to_dict("records"),
        "lasso_item_count_metrics": lasso.to_dict("records"),
        "random_rawsum_summary": random_sum.to_dict("records"),
        "triage": triage.to_dict("records"),
        "selective_escalation": selective.to_dict("records"),
        "generated_figures": generated_figures,
        "journal_upload_figures": journal_upload_figures,
    }
    with open(args.out / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Created PNG previews and editable SVG/PDF sources in", F)
    print("Created one 600-dpi LZW TIFF per figure in", upload_dir)
    print("Figure font:", FIGURE_FONT_FAMILY, "| genuine Arial available:", TRUE_ARIAL_AVAILABLE)


if __name__ == "__main__":
    main()
