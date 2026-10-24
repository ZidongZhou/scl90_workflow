#!/usr/bin/env python3
"""Reproducible analysis for the PLOS ONE SCL-90 follow-up triage study.

The script reads the public item-level CSV, applies quality checks, constructs
full-length SCL-90-derived screening labels, fits short item-set models, evaluates
location-code-based hold-out validation, computes random raw-score item-set baselines, and
exports the deterministic model-evaluation tables needed for the manuscript figures.

By default, the LASSO/L1 ranking used for the operational 15-item example is read
from an archived ranking file under code/static/. This avoids solver-version drift
in sparsity paths while preserving the exact item order used in the manuscript.
The current-environment LASSO ranking can be regenerated with
--write-algorithmic-lasso-ranking; it is exported separately and is not used to
replace the archived operational ranking unless explicitly requested.
"""
from __future__ import annotations
import argparse, json, platform, shutil, sys, warnings
from pathlib import Path
from typing import Sequence
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 20260610
ITEMS = [f"S{i}" for i in range(1, 91)]

# Standard SCL-90 domain grouping used only to enforce content coverage in the
# order-assumed domain-balanced sensitivity item sets. Item wording is not reproduced.
SCL90_DOMAINS = {
    "Somatization": [1, 4, 12, 27, 40, 42, 48, 49, 52, 53, 56, 58],
    "Obsessive-compulsive": [3, 9, 10, 28, 38, 45, 46, 51, 55, 65],
    "Interpersonal sensitivity": [6, 21, 34, 36, 37, 41, 61, 69, 73],
    "Depression": [5, 14, 15, 20, 22, 26, 29, 30, 31, 32, 54, 71, 79],
    "Anxiety": [2, 17, 23, 33, 39, 57, 72, 78, 80, 86],
    "Hostility": [11, 24, 63, 67, 74, 81],
    "Phobic anxiety": [13, 25, 47, 50, 70, 75, 82],
    "Paranoid ideation": [8, 18, 43, 68, 76, 83],
    "Psychoticism": [7, 16, 35, 62, 77, 84, 85, 87, 88, 90],
    "Additional items": [19, 44, 59, 60, 64, 66, 89],
}
ITEM_TO_DOMAIN = {f"S{i}": domain for domain, nums in SCL90_DOMAINS.items() for i in nums}

# Keep reviewer logs clean across scikit-learn minor-version differences.
# The locked environment uses scikit-learn 1.8.0; these narrowly targeted
# filters avoid harmless cross-version L1/elastic-net deprecation chatter if a
# reviewer runs the scripts in a newer existing environment. They do not silence
# other warnings.
warnings.filterwarnings(
    "ignore",
    message=r".*penalty.*deprecated.*",
    category=FutureWarning,
    module=r"sklearn.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*Inconsistent values: penalty=l1 with l1_ratio.*",
    category=UserWarning,
    module=r"sklearn.*",
)


def log_step(message: str) -> None:
    """Emit progress messages for long-running stages."""
    print(f"[analysis] {message}", flush=True)


def ensure_dirs(base: Path) -> tuple[Path, Path]:
    tables = base / "tables"; figures = base / "figures"
    tables.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    return tables, figures


def validate_schema(df: pd.DataFrame, path: Path) -> None:
    """Fail fast on file-structure problems while leaving row-level QC to load_and_qc.

    The public CSV contains one invalid respondent record in the submitted analysis.
    Therefore, this function checks file integrity, required variables, numeric/integer
    coding, and the location mapping, but does not reject rows solely because an item
    response is outside the permitted 0-4 range. Such rows are excluded and reported
    by the predefined quality-control step below.
    """
    required = ["location", "gender", "total", *ITEMS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    if len(df) != 14372:
        warnings.warn(f"Expected 14,372 rows from Mendeley Data V1; found {len(df):,} rows in {path}.", RuntimeWarning)
    if df[required].isna().any().any():
        bad_cols = df[required].columns[df[required].isna().any()].tolist()
        raise ValueError(f"Missing values detected in required columns: {bad_cols}")
    for col in required:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column {col} must be numeric/integer-coded; observed dtype {df[col].dtype}")
        vals = df[col].to_numpy()
        if not np.all(np.isfinite(vals)):
            raise ValueError(f"Column {col} contains non-finite values")
        if not np.all(np.equal(vals, np.round(vals))):
            raise ValueError(f"Column {col} must be integer-coded")
    locations = set(df["location"].astype(int).unique().tolist())
    if locations != {1, 2}:
        raise ValueError(f"Expected location codes exactly {{1, 2}}; observed {sorted(locations)}")


def load_and_qc(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    validate_schema(df, path)
    item_sum = df[ITEMS].sum(axis=1)
    invalid = (
        df[ITEMS].lt(0).any(axis=1)
        | df[ITEMS].gt(4).any(axis=1)
        | ~df["gender"].isin([1, 2])
        | ((item_sum + 90) != df["total"])
    )
    excluded = df.loc[invalid].copy()
    d = df.loc[~invalid].copy().reset_index(drop=True)
    d["item_sum_0_4"] = d[ITEMS].sum(axis=1)
    d["label_ge160"] = (d["total"] >= 160).astype(int)
    d["label_gt160"] = (d["total"] > 160).astype(int)
    region_map = {1: "Shandong", 2: "Hunan"}
    d["region"] = d["location"].map(region_map)
    if d["region"].isna().any():
        raise ValueError("Unmapped location code after QC; expected location=1 for Shandong and location=2 for Hunan")
    d["all_same"] = d[ITEMS].nunique(axis=1).eq(1)
    d["all_zero"] = d[ITEMS].sum(axis=1).eq(0)
    d["duplicate_item_pattern"] = d.duplicated(ITEMS, keep=False)
    return d, excluded


def sample_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in [
        ("Overall", d),
        ("Mapped Hunan code group", d[d.region == "Hunan"]),
        ("Mapped Shandong code group", d[d.region == "Shandong"]),
    ]:
        rows.append({
            "sample": name,
            "n": len(g),
            "gender_code_1_n": int((g.gender == 1).sum()),
            "gender_code_1_percent": 100 * (g.gender == 1).mean(),
            "gender_code_2_n": int((g.gender == 2).sum()),
            "gender_code_2_percent": 100 * (g.gender == 2).mean(),
            "screen_positive_n": int(g.label_ge160.sum()),
            "screen_positive_percent": 100 * g.label_ge160.mean(),
            "total_mean": g.total.mean(),
            "total_sd": g.total.std(ddof=1),
            "all_same_n": int(g.all_same.sum()),
            "all_zero_n": int(g.all_zero.sum()),
            "duplicate_item_pattern_n": int(g.duplicate_item_pattern.sum()),
        })
    return pd.DataFrame(rows)


def outcome_table(d: pd.DataFrame, main_items: Sequence[str] | None = None) -> pd.DataFrame:
    q90 = d["item_sum_0_4"].quantile(0.90, interpolation="lower")
    specs = [
        ("label_ge160", "SCL-90 total score >=160; item sum >=70", d.label_ge160),
        ("label_gt160", "SCL-90 total score >160; item sum >70", d.label_gt160),
        ("label_top10_full_score", "Top decile of full 90-item symptom burden", (d.item_sum_0_4 >= q90).astype(int)),
        ("label_ge200", "SCL-90 total score >=200; high symptom-burden sensitivity analysis", (d.total >= 200).astype(int)),
    ]
    if main_items is not None:
        rem_items = [i for i in ITEMS if i not in main_items]
        specs.append(("remaining_75_item_label_ge59", "Remaining 75-item raw sum >=59 after excluding the 15-item operational set", (d[rem_items].sum(axis=1) >= 59).astype(int)))
    rows = []
    for key, definition, y in specs:
        for sample, mask in [
            ("Overall", np.ones(len(d), dtype=bool)),
            ("Mapped Hunan code group", d.region.eq("Hunan")),
            ("Mapped Shandong code group", d.region.eq("Shandong")),
        ]:
            yy = y[mask]
            rows.append({"outcome": key, "definition": definition, "sample": sample, "positive_n": int(yy.sum()), "positive_percent": 100*yy.mean()})
    return pd.DataFrame(rows)


def fit_logit(X_train: np.ndarray, y_train: np.ndarray, weighted: bool = True) -> LogisticRegression:
    return LogisticRegression(C=1.0, solver="lbfgs", class_weight="balanced" if weighted else None, max_iter=2000, random_state=RANDOM_STATE).fit(X_train, y_train)


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Fast Youden-index threshold using vectorized ROC coordinates.

    Earlier versions evaluated 501 candidate quantiles with repeated
    ``confusion_matrix`` calls. That is correct but unnecessarily slow in the
    end-to-end reviewer route because the function is called repeatedly across
    item-set, subgroup, sensitivity, and nested-validation models.
    """
    if len(np.unique(y)) < 2:
        return float(np.nanmedian(p))
    fpr, tpr, thresholds = roc_curve(y, p)
    valid = np.isfinite(thresholds)
    if not np.any(valid):
        return 0.5
    j = tpr[valid] - fpr[valid]
    return float(thresholds[valid][int(np.nanargmax(j))])


def calibration_metrics(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    eps = 1e-6
    lp = np.log(np.clip(p, eps, 1-eps) / np.clip(1-p, eps, 1-eps))
    try:
        lr = LogisticRegression(C=1e6, solver="liblinear", max_iter=200).fit(lp.reshape(-1,1), y)
        return float(lr.intercept_[0]), float(lr.coef_[0,0])
    except Exception:
        return np.nan, np.nan


def deterministic_descending_order(scores: np.ndarray) -> np.ndarray:
    """Sort scores descending with original row order as an explicit tie-break.

    Raw-sum comparators contain many ties. NumPy's default quicksort is not a
    stable cross-version tie policy, so an explicit secondary row-index key is
    required for iteration-level reproducibility of top-k metrics.
    """
    scores = np.asarray(scores, dtype=float)
    return np.lexsort((np.arange(len(scores), dtype=int), -scores))


def topk_metrics(y: np.ndarray, p: np.ndarray, ks: Sequence[int] = (5,10,15,20,25,30)) -> dict:
    order = deterministic_descending_order(p); out = {}; npos = y.sum(); prevalence = y.mean()
    for k in ks:
        n_top = max(1, int(round(len(y) * k / 100)))
        idx = order[:n_top]
        prec = y[idx].mean()
        out[f"top{k}_recall"] = y[idx].sum() / npos if npos else np.nan
        out[f"top{k}_precision"] = prec
        out[f"top{k}_lift"] = prec / prevalence if prevalence else np.nan
    return out


def evaluate(y: np.ndarray, p: np.ndarray, threshold: float | None = None) -> dict:
    out = {"n": len(y), "positive_n": int(y.sum()), "prevalence": float(y.mean()), "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p)}
    if np.nanmin(p) >= 0 and np.nanmax(p) <= 1:
        out["brier"] = brier_score_loss(y, p)
        ci, cs = calibration_metrics(y, p); out["calibration_intercept"] = ci; out["calibration_slope"] = cs
    else:
        out["brier"] = np.nan; out["calibration_intercept"] = np.nan; out["calibration_slope"] = np.nan
    if threshold is not None:
        pred = (p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
        out.update({"sensitivity": tp/(tp+fn) if tp+fn else np.nan, "specificity": tn/(tn+fp) if tn+fp else np.nan, "ppv": tp/(tp+fp) if tp+fp else np.nan, "npv": tn/(tn+fn) if tn+fn else np.nan, "predicted_positive_rate": pred.mean(), "threshold": threshold})
    out.update(topk_metrics(y, p))
    return out


def read_archived_lasso(static_dir: Path) -> pd.DataFrame:
    p = static_dir / "lasso_rank_hunan.csv"
    r = pd.read_csv(p)
    if "importance" not in r.columns:
        if "abs_score" in r.columns:
            r = r.rename(columns={"abs_score":"importance"})
        elif "score" in r.columns:
            r["importance"] = r["score"].abs()
    return r[["item", "importance"]].sort_values("importance", ascending=False).reset_index(drop=True)


def rank_items_correlation(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    vals = [abs(np.corrcoef(X[c], y)[0,1]) for c in X.columns]
    return pd.DataFrame({"item": X.columns, "importance": vals}).sort_values("importance", ascending=False).reset_index(drop=True)


def rank_items_mi(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    vals = mutual_info_classif(X, y, discrete_features=True, random_state=RANDOM_STATE)
    return pd.DataFrame({"item": X.columns, "importance": vals}).sort_values("importance", ascending=False).reset_index(drop=True)


def rank_items_elastic_net_sgd(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    scaler = StandardScaler().fit(X); Xs = scaler.transform(X)
    clf = SGDClassifier(loss="log_loss", penalty="elasticnet", alpha=0.0002, l1_ratio=0.5, class_weight="balanced", max_iter=3000, random_state=RANDOM_STATE)
    clf.fit(Xs, y)
    vals = np.abs(clf.coef_[0])
    return pd.DataFrame({"item": X.columns, "importance": vals}).sort_values("importance", ascending=False).reset_index(drop=True)


def rank_items_lasso_algorithmic(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    scaler = StandardScaler().fit(X); Xs = scaler.transform(X)
    clf = LogisticRegression(penalty="l1", C=0.2, solver="liblinear", class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)
    clf.fit(Xs, y)
    vals = np.abs(clf.coef_[0])
    return pd.DataFrame({"item": X.columns, "importance": vals}).sort_values("importance", ascending=False).reset_index(drop=True)


def safe_filename(name: str) -> str:
    mapping = {
        "LASSO/L1 ranking": "lasso_l1_item_ranking_hunan_all90.csv",
        "Correlation ranking": "correlation_item_ranking_hunan_all90.csv",
        "Mutual information ranking": "mutual_information_item_ranking_hunan_all90.csv",
        "Elastic net ranking (SGD)": "elastic_net_sgd_item_ranking_hunan_all90.csv",
    }
    return mapping[name]


def fit_eval_itemset(train: pd.DataFrame, test: pd.DataFrame, items: Sequence[str], label: str = "label_ge160", weighted: bool = True):
    scaler = StandardScaler().fit(train[list(items)])
    Xtr, Xte = scaler.transform(train[list(items)]), scaler.transform(test[list(items)])
    ytr, yte = train[label].to_numpy(), test[label].to_numpy()
    model = fit_logit(Xtr, ytr, weighted=weighted)
    p_tr = model.predict_proba(Xtr)[:,1]; p_te = model.predict_proba(Xte)[:,1]
    thr = youden_threshold(ytr, p_tr)
    return evaluate(yte, p_te, thr), model, scaler, thr, p_tr, p_te


def random_rawsum_baseline(test: pd.DataFrame, item_counts: Sequence[int], repeats: int, seed: int):
    rng = np.random.default_rng(seed); rows = []; yte = test.label_ge160.to_numpy()
    for k in item_counts:
        for b in range(repeats):
            sel = sorted(rng.choice(ITEMS, size=k, replace=False).tolist(), key=lambda s: int(s[1:]))
            score = test[sel].sum(axis=1).to_numpy()
            ev = evaluate(yte, score, None)
            rows.append({"item_count": k, "iteration": b+1, "selected_items": ",".join(sel),
                         "roc_auc": ev["roc_auc"], "pr_auc": ev["pr_auc"],
                         "top10_recall": ev["top10_recall"], "top10_precision": ev["top10_precision"],
                         "top20_recall": ev["top20_recall"], "top20_precision": ev["top20_precision"]})
    dist = pd.DataFrame(rows)
    qs = lambda q: (lambda s: s.quantile(q))
    summ = dist.groupby("item_count").agg(
        n_repeats=("iteration","count"),
        roc_auc_mean=("roc_auc","mean"), roc_auc_sd=("roc_auc","std"), roc_auc_p025=("roc_auc",qs(.025)), roc_auc_p975=("roc_auc",qs(.975)),
        pr_auc_mean=("pr_auc","mean"), pr_auc_sd=("pr_auc","std"), pr_auc_p025=("pr_auc",qs(.025)), pr_auc_p975=("pr_auc",qs(.975)),
        top10_recall_mean=("top10_recall","mean"), top10_recall_sd=("top10_recall","std"), top10_recall_p025=("top10_recall",qs(.025)), top10_recall_p975=("top10_recall",qs(.975)),
        top10_precision_mean=("top10_precision","mean"), top10_precision_sd=("top10_precision","std"), top10_precision_p025=("top10_precision",qs(.025)), top10_precision_p975=("top10_precision",qs(.975)),
        top20_recall_mean=("top20_recall","mean"), top20_recall_sd=("top20_recall","std"),
        top20_precision_mean=("top20_precision","mean"), top20_precision_sd=("top20_precision","std"),
    ).reset_index()
    return dist, summ


def bootstrap_ci(y: np.ndarray, p: np.ndarray, thr: float, repeats: int = 500, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Nonparametric bootstrap CIs for primary validation and top-k triage metrics.

    Passing ``--bootstrap-ci 0`` intentionally skips resampling while still
    exporting the expected metric rows with empty CI columns. This makes quick
    reviewer smoke-runs possible without breaking downstream SI-table creation.
    """
    ks = (5, 10, 15, 20, 25, 30)
    base_metrics = ["roc_auc", "pr_auc", "brier", "sensitivity", "specificity", "ppv", "npv"]
    topk_metrics_names = [f"top{k}_{suffix}" for k in ks for suffix in ("recall", "precision", "lift")]
    metrics_template = base_metrics + topk_metrics_names
    if repeats <= 0:
        return pd.DataFrame({
            "metric": metrics_template,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "bootstrap_repeats": 0,
        })

    rng = np.random.default_rng(seed); rows = []
    n = len(y)
    for _ in range(repeats):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2:
            continue
        pred = (pp >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(yy, pred, labels=[0,1]).ravel()
        row = {
            "roc_auc": roc_auc_score(yy, pp),
            "pr_auc": average_precision_score(yy, pp),
            "brier": brier_score_loss(yy, pp),
            "sensitivity": tp/(tp+fn) if tp+fn else np.nan,
            "specificity": tn/(tn+fp) if tn+fp else np.nan,
            "ppv": tp/(tp+fp) if tp+fp else np.nan,
            "npv": tn/(tn+fn) if tn+fn else np.nan,
        }
        prevalence = yy.mean()
        order = deterministic_descending_order(pp)
        for k in ks:
            n_top = max(1, int(round(n * k / 100)))
            top = order[:n_top]
            precision = yy[top].mean()
            row[f"top{k}_recall"] = yy[top].sum() / yy.sum() if yy.sum() else np.nan
            row[f"top{k}_precision"] = precision
            row[f"top{k}_lift"] = precision / prevalence if prevalence else np.nan
        rows.append(row)
    bd = pd.DataFrame(rows)
    if bd.empty:
        return pd.DataFrame({
            "metric": metrics_template,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "bootstrap_repeats": 0,
        })
    metrics = list(bd.columns)
    return pd.DataFrame({"metric": metrics, "ci_low": [bd[m].quantile(.025) for m in metrics], "ci_high": [bd[m].quantile(.975) for m in metrics], "bootstrap_repeats": len(bd)})


def data_quality_exclusion_sensitivity(d: pd.DataFrame, items: Sequence[str]) -> pd.DataFrame:
    """Refit the fixed 15-item operational model after excluding potentially low-quality response patterns.

    The item set is kept fixed to the archived operational 15 items. This sensitivity
    analysis tests whether the main Hunan-to-Shandong discrimination and top-k triage
    findings are driven by all-same responses or repeated 90-item response patterns.
    """
    non_all_same = ~d["all_same"]
    dedup_after_excluding_all_same = pd.Series(False, index=d.index)
    dedup_after_excluding_all_same.loc[non_all_same] = ~d.loc[non_all_same].duplicated(ITEMS, keep="first")
    strategies = [
        ("Main analytic sample", np.ones(len(d), dtype=bool)),
        ("Exclude all-same response patterns", non_all_same.to_numpy()),
        ("Deduplicate 90-item response patterns: retain first occurrence", ~d.duplicated(ITEMS, keep="first").to_numpy()),
        ("Exclude all-same and deduplicate 90-item patterns", dedup_after_excluding_all_same.to_numpy()),
    ]
    rows = []
    for name, mask in strategies:
        subset = d.loc[mask].copy().reset_index(drop=True)
        train = subset[subset.region == "Hunan"].copy().reset_index(drop=True)
        test = subset[subset.region == "Shandong"].copy().reset_index(drop=True)
        if len(train) == 0 or len(test) == 0 or train.label_ge160.nunique() < 2 or test.label_ge160.nunique() < 2:
            rows.append({"analysis": name, "note": "insufficient data after exclusion"})
            continue
        ev, _, _, _, _, _ = fit_eval_itemset(train, test, items, label="label_ge160", weighted=True)
        prevalence = ev.get("prevalence", np.nan)
        theoretical_top10 = min(1.0, 0.10 / prevalence) if prevalence and not np.isnan(prevalence) else np.nan
        rows.append({
            "analysis": name,
            "shandong_n": int(ev.get("n", len(test))),
            "estimated_shandong_screen_positive_n": int(ev.get("positive_n", np.nan)),
            "estimated_shandong_screen_positive_prevalence_percent": round(100 * prevalence, 1),
            "roc_auc": round(ev.get("roc_auc", np.nan), 3),
            "pr_auc": round(ev.get("pr_auc", np.nan), 3),
            "top10_recall_percent": round(100 * ev.get("top10_recall", np.nan), 1),
            "top10_screen_positive_rate_percent": round(100 * ev.get("top10_precision", np.nan), 1),
            "theoretical_max_top10_recall_percent": round(100 * theoretical_top10, 1),
            "note": "Submitted SI-ready summary; fixed 15-item operational example refit under the stated response-quality rule",
        })
    return pd.DataFrame(rows)


def topk_bootstrap_ci_table(point_metrics: dict, ci_table: pd.DataFrame, n_validation: int, ks: Sequence[int] = (5,10,15,20,25,30)) -> pd.DataFrame:
    """Create an SI-ready top-k triage CI table from point estimates and bootstrap CIs.

    Values are reported in percentages for direct comparison with Table 5.
    The table always includes all submitted capacity levels (5%-30%).
    """
    ci = ci_table.set_index("metric")
    rows = []
    for k in ks:
        rows.append({
            "capacity_percent": k,
            "students_reviewed": int(round(n_validation * k / 100)),
            "coverage_recall_point_percent": 100 * point_metrics.get(f"top{k}_recall", np.nan),
            "coverage_recall_95ci_low_percent": 100 * ci.loc[f"top{k}_recall", "ci_low"] if f"top{k}_recall" in ci.index else np.nan,
            "coverage_recall_95ci_high_percent": 100 * ci.loc[f"top{k}_recall", "ci_high"] if f"top{k}_recall" in ci.index else np.nan,
            "screen_positive_rate_point_percent": 100 * point_metrics.get(f"top{k}_precision", np.nan),
            "screen_positive_rate_95ci_low_percent": 100 * ci.loc[f"top{k}_precision", "ci_low"] if f"top{k}_precision" in ci.index else np.nan,
            "screen_positive_rate_95ci_high_percent": 100 * ci.loc[f"top{k}_precision", "ci_high"] if f"top{k}_precision" in ci.index else np.nan,
            "lift_point": point_metrics.get(f"top{k}_lift"),
            "lift_95ci_low": ci.loc[f"top{k}_lift", "ci_low"] if f"top{k}_lift" in ci.index else np.nan,
            "lift_95ci_high": ci.loc[f"top{k}_lift", "ci_high"] if f"top{k}_lift" in ci.index else np.nan,
            "bootstrap_repeats": int(ci_table["bootstrap_repeats"].max()) if "bootstrap_repeats" in ci_table else np.nan,
            "note": "Nonparametric validation bootstrap conditional on the fixed 15-item operational example"
        })
    return pd.DataFrame(rows)


def calibration_bins(y: np.ndarray, p: np.ndarray, label: str, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    out = df.groupby("bin", observed=True).agg(mean_predicted_probability=("p","mean"), observed_screen_positive_rate=("y","mean"), n=("y","size")).reset_index(drop=True)
    out.insert(0, "model", label)
    return out


def pooled_cv(d: pd.DataFrame, items: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = d.label_ge160.to_numpy(); region = d.region.astype(str).to_numpy()
    strat = np.array([f"{yy}_{rr}" for yy, rr in zip(y, region)])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(d[items], strat), 1):
        scaler = StandardScaler().fit(d.iloc[tr][list(items)])
        clf = fit_logit(scaler.transform(d.iloc[tr][list(items)]), y[tr], weighted=True)
        pte = clf.predict_proba(scaler.transform(d.iloc[te][list(items)]))[:,1]
        rows.append({"fold": fold, **evaluate(y[te], pte, None)})
    folds = pd.DataFrame(rows)
    summ = folds.drop(columns=["fold"]).agg(['mean','std']).T.reset_index().rename(columns={"index":"metric"})
    return folds, summ



def domain_balanced_items(train: pd.DataFrame, fill_ranking: pd.DataFrame, n_items: int = 15) -> tuple[list[str], pd.DataFrame]:
    """Select at least one item from each SCL-90 content domain, then fill by ranking.

    The design is intentionally deterministic and transparent: the highest-ranked
    item within each standard domain is selected first, and remaining slots are
    filled by the same Hunan-development ranking after removing duplicates. This
    is an order-assumed domain-balanced sensitivity analysis, not a new validated short form.
    """
    selected: list[str] = []
    rows = []
    ranking = fill_ranking.copy().reset_index(drop=True)
    for domain, nums in SCL90_DOMAINS.items():
        candidates = [f"S{i}" for i in nums]
        sub = ranking[ranking["item"].isin(candidates)].copy()
        if sub.empty:
            continue
        item = str(sub.iloc[0]["item"])
        selected.append(item)
        rows.append({"selection_stage": "domain anchor", "domain": domain, "item": item, "rank_importance": float(sub.iloc[0]["importance"])})
    for item in ranking["item"].tolist():
        if len(selected) >= n_items:
            break
        if item not in selected:
            selected.append(item)
            rows.append({"selection_stage": "ranked fill", "domain": ITEM_TO_DOMAIN.get(item, "Unmapped"), "item": item, "rank_importance": float(ranking.loc[ranking.item == item, "importance"].iloc[0])})
    return selected[:n_items], pd.DataFrame(rows[:n_items])


def selective_escalation_threshold_grid(train: pd.DataFrame, items: Sequence[str], low_npv_min: float = 0.995, high_ppv_min: float = 0.960, accepted_error_max: float = 0.008) -> pd.DataFrame:
    """Search simple raw-sum low/high thresholds in the development sample.

    Thresholds are selected only from the development sample. The default predictive-value
    constraints were chosen to keep the low zone very conservative while requiring
    a high-zone PPV compatible with first-wave follow-up prioritization.
    """
    y = train.label_ge160.to_numpy()
    raw = train[list(items)].sum(axis=1).to_numpy()
    rows = []
    max_score = int(raw.max())
    for low_cut in range(0, max_score + 1):
        for high_cut in range(low_cut + 1, max_score + 1):
            low = raw <= low_cut
            high = raw >= high_cut
            accepted = low | high
            if low.sum() == 0 or high.sum() == 0:
                continue
            low_npv = 1 - y[low].mean()
            high_ppv = y[high].mean()
            accepted_error_n = int(y[low].sum() + (1 - y[high]).sum())
            accepted_error_rate_all = accepted_error_n / len(y)
            accepted_error_rate_accepted = accepted_error_n / accepted.sum()
            if low_npv >= low_npv_min and high_ppv >= high_ppv_min and accepted_error_rate_all <= accepted_error_max:
                avg_items = (15 * accepted.sum() + 90 * (~accepted).sum()) / len(y)
                rows.append({
                    "low_cut_rawsum_le": low_cut,
                    "high_cut_rawsum_ge": high_cut,
                    "accepted_proportion": accepted.mean(),
                    "escalation_proportion": (~accepted).mean(),
                    "low_zone_n": int(low.sum()),
                    "middle_zone_n": int((~accepted).sum()),
                    "high_zone_n": int(high.sum()),
                    "low_zone_npv": low_npv,
                    "high_zone_ppv": high_ppv,
                    "accepted_error_n": accepted_error_n,
                    "accepted_error_rate_all": accepted_error_rate_all,
                    "accepted_error_rate_accepted": accepted_error_rate_accepted,
                    "average_item_burden": avg_items,
                    "burden_reduction_vs_90": (90 - avg_items) / 90,
                    "constraint_low_npv_min": low_npv_min,
                    "constraint_high_ppv_min": high_ppv_min,
                    "constraint_accepted_error_max": accepted_error_max,
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["accepted_proportion", "low_zone_npv", "high_zone_ppv"], ascending=[False, False, False]).reset_index(drop=True)
    return out


def selective_escalation_metrics(df: pd.DataFrame, items: Sequence[str], low_cut: int, high_cut: int, sample: str) -> dict:
    y = df.label_ge160.to_numpy()
    raw = df[list(items)].sum(axis=1).to_numpy()
    low = raw <= low_cut
    high = raw >= high_cut
    middle = ~(low | high)
    accepted = low | high
    false_neg_low = int(y[low].sum())
    false_pos_high = int((1 - y[high]).sum())
    err = false_neg_low + false_pos_high
    avg_items = (15 * accepted.sum() + 90 * middle.sum()) / len(y)
    return {
        "sample": sample,
        "n": int(len(y)),
        "positive_n": int(y.sum()),
        "prevalence": float(y.mean()),
        "low_cut_rawsum_le": int(low_cut),
        "high_cut_rawsum_ge": int(high_cut),
        "low_zone_n": int(low.sum()),
        "middle_escalation_n": int(middle.sum()),
        "high_zone_n": int(high.sum()),
        "accepted_n": int(accepted.sum()),
        "accepted_proportion": float(accepted.mean()),
        "escalation_proportion": float(middle.mean()),
        "low_zone_npv": float(1 - y[low].mean()) if low.sum() else np.nan,
        "high_zone_ppv": float(y[high].mean()) if high.sum() else np.nan,
        "false_negative_low_zone_n": false_neg_low,
        "false_positive_high_zone_n": false_pos_high,
        "accepted_error_n": int(err),
        "accepted_error_rate_all": float(err / len(y)),
        "accepted_error_rate_accepted": float(err / accepted.sum()) if accepted.sum() else np.nan,
        "average_item_burden": float(avg_items),
        "burden_reduction_vs_90": float((90 - avg_items) / 90),
        "workflow_note": "15-item raw-sum first stage: low zone no immediate full form; middle zone completes full SCL-90; high zone prioritized for follow-up",
    }


def selective_escalation_error_profile(df: pd.DataFrame, items: Sequence[str], low_cut: int, high_cut: int, sample: str) -> pd.DataFrame:
    """Summarize near-threshold error profiles without writing individual records."""
    y = df.label_ge160.to_numpy()
    raw = df[list(items)].sum(axis=1).to_numpy()
    low_fn = df.loc[(raw <= low_cut) & (y == 1), "total"]
    high_fp = df.loc[(raw >= high_cut) & (y == 0), "total"]
    rows = []
    for error_type, values in [
        ("low-zone false negatives", low_fn),
        ("high-zone false positives", high_fp),
    ]:
        rows.append({
            "sample": sample,
            "error_type": error_type,
            "n": int(len(values)),
            "total_min": float(values.min()) if len(values) else np.nan,
            "total_median": float(values.median()) if len(values) else np.nan,
            "total_max": float(values.max()) if len(values) else np.nan,
            "total_ge200_n": int((values >= 200).sum()) if len(values) else 0,
            "note": "Aggregate error-profile summary only; individual-level records are not written by default.",
        })
    return pd.DataFrame(rows)


def selective_escalation_bootstrap_ci(df: pd.DataFrame, items: Sequence[str], low_cut: int, high_cut: int, repeats: int = 500, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    n = len(df)
    metrics = ["accepted_proportion", "escalation_proportion", "low_zone_npv", "high_zone_ppv", "accepted_error_rate_all", "accepted_error_rate_accepted", "average_item_burden", "burden_reduction_vs_90"]
    if repeats <= 0:
        return pd.DataFrame({"metric": metrics, "ci_low": np.nan, "ci_high": np.nan, "bootstrap_repeats": 0})
    for _ in range(repeats):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        m = selective_escalation_metrics(df.iloc[idx].reset_index(drop=True), items, low_cut, high_cut, "bootstrap")
        rows.append(m)
    bd = pd.DataFrame(rows)
    return pd.DataFrame({"metric": metrics, "ci_low": [bd[m].quantile(.025) for m in metrics], "ci_high": [bd[m].quantile(.975) for m in metrics], "bootstrap_repeats": repeats})


def response_quality_restricted_selective_escalation_sensitivity(d: pd.DataFrame, items: Sequence[str]) -> pd.DataFrame:
    """Rerun threshold selection and hold-out workflow evaluation after response-quality restrictions.

    This table is designed for reviewer scrutiny of the main deployment claim:
    whether the selective full-form escalation workflow remains acceptable after
    excluding all-same responses and/or applying retrospective administrative
    handling of duplicated 90-item response patterns.
    """
    non_all_same = ~d["all_same"]
    dedup_after_excluding_all_same = pd.Series(False, index=d.index)
    dedup_after_excluding_all_same.loc[non_all_same] = ~d.loc[non_all_same].duplicated(ITEMS, keep="first")
    strategies = [
        ("Main analytic sample", np.ones(len(d), dtype=bool)),
        ("Exclude all-same response patterns", non_all_same.to_numpy()),
        ("Deduplicate 90-item response patterns: retain first occurrence", ~d.duplicated(ITEMS, keep="first").to_numpy()),
        ("Exclude all-same and deduplicate 90-item patterns", dedup_after_excluding_all_same.to_numpy()),
    ]
    rows = []
    for strategy, mask in strategies:
        subset = d.loc[mask].copy().reset_index(drop=True)
        train = subset[subset.region == "Hunan"].copy().reset_index(drop=True)
        test = subset[subset.region == "Shandong"].copy().reset_index(drop=True)
        row = {
            "response_quality_restriction": strategy,
            "hunan_development_n": int(len(train)),
            "shandong_holdout_n": int(len(test)),
            "workflow_note": "Thresholds are reselected within the response-quality-restricted mapped Hunan code group and evaluated in the corresponding mapped Shandong code group.",
        }
        if len(train) == 0 or len(test) == 0 or train.label_ge160.nunique() < 2 or test.label_ge160.nunique() < 2:
            row["note"] = "insufficient data after response-quality restriction"
            rows.append(row)
            continue
        grid = selective_escalation_threshold_grid(train, items)
        if grid.empty:
            row["note"] = "no threshold pair met the prespecified development constraints after response-quality restriction"
            rows.append(row)
            continue
        low_cut = int(grid.iloc[0]["low_cut_rawsum_le"])
        high_cut = int(grid.iloc[0]["high_cut_rawsum_ge"])
        dev_metrics = selective_escalation_metrics(train, items, low_cut, high_cut, "mapped Hunan code group after response-pattern audit")
        hold_metrics = selective_escalation_metrics(test, items, low_cut, high_cut, "mapped Shandong code group after response-pattern audit")
        row.update({
            "low_cut_rawsum_le": low_cut,
            "high_cut_rawsum_ge": high_cut,
            "development_low_zone_npv": dev_metrics["low_zone_npv"],
            "development_high_zone_ppv": dev_metrics["high_zone_ppv"],
            "development_accepted_error_rate_all": dev_metrics["accepted_error_rate_all"],
            "holdout_low_zone_npv": hold_metrics["low_zone_npv"],
            "holdout_high_zone_ppv": hold_metrics["high_zone_ppv"],
            "holdout_accepted_error_rate_all": hold_metrics["accepted_error_rate_all"],
            "holdout_accepted_error_rate_accepted": hold_metrics["accepted_error_rate_accepted"],
            "holdout_escalation_proportion": hold_metrics["escalation_proportion"],
            "holdout_average_item_burden": hold_metrics["average_item_burden"],
            "holdout_burden_reduction_vs_90": hold_metrics["burden_reduction_vs_90"],
            "holdout_false_negative_low_zone_n": hold_metrics["false_negative_low_zone_n"],
            "holdout_false_positive_high_zone_n": hold_metrics["false_positive_high_zone_n"],
            "deployment_caution": "These analyses are response-quality-restricted sensitivity analyses. Full-90 duplicate-pattern handling is retrospective/administrative and should not be presented as a first-stage observable deployment gate.",
        })
        rows.append(row)
    return pd.DataFrame(rows)


def selective_escalation_threshold_tradeoff(train: pd.DataFrame, test: pd.DataFrame, items: Sequence[str], base_low: int = 13, base_high: int = 24) -> pd.DataFrame:
    """Show local threshold trade-offs around the selected workflow parameters.

    The table does not optimize thresholds in the hold-out group. It documents
    how nearby low-zone and high-zone cut-offs change NPV, PPV, accepted error,
    escalation, and item burden, so the selected pair is interpreted as a
    retrospective workflow parameter rather than a transportable cut-off.
    """
    rows = []
    configs = []
    for low_cut in [11, 12, 13, 14]:
        configs.append(("low-threshold trade-off with high threshold fixed", low_cut, base_high))
    for high_cut in [23, 24, 25]:
        configs.append(("high-threshold trade-off with low threshold fixed", base_low, high_cut))
    for analysis, low_cut, high_cut in configs:
        dev = selective_escalation_metrics(train, items, low_cut, high_cut, "mapped Hunan development group")
        hold = selective_escalation_metrics(test, items, low_cut, high_cut, "mapped Shandong hold-out group")
        rows.append({
            "analysis": analysis,
            "low_cut_rawsum_le": low_cut,
            "high_cut_rawsum_ge": high_cut,
            "is_selected_pair": bool(low_cut == base_low and high_cut == base_high),
            "development_low_zone_npv": dev["low_zone_npv"],
            "development_high_zone_ppv": dev["high_zone_ppv"],
            "development_accepted_error_rate_all": dev["accepted_error_rate_all"],
            "development_escalation_proportion": dev["escalation_proportion"],
            "development_average_item_burden": dev["average_item_burden"],
            "holdout_low_zone_npv": hold["low_zone_npv"],
            "holdout_high_zone_ppv": hold["high_zone_ppv"],
            "holdout_accepted_error_rate_all": hold["accepted_error_rate_all"],
            "holdout_escalation_proportion": hold["escalation_proportion"],
            "holdout_average_item_burden": hold["average_item_burden"],
            "holdout_false_negative_low_zone_n": hold["false_negative_low_zone_n"],
            "holdout_false_positive_high_zone_n": hold["false_positive_high_zone_n"],
            "interpretation": "Nearby thresholds trade lower burden against low-zone NPV/high-zone PPV; the selected pair is not a transportable cut-off without local recalibration.",
        })
    return pd.DataFrame(rows)


# The response-quality-restricted sensitivity implementation is defined above.
# No recursive compatibility wrapper is needed here.


def retrospective_response_quality_audit_simulation(d: pd.DataFrame, items: Sequence[str], low_cut: int = 13, high_cut: int = 24) -> pd.DataFrame:
    """Retrospective response-quality audit based on full-form response patterns.

    All-same, all-zero, or duplicate 90-item patterns are routed to quality
    audit/retest/full-form review in this retrospective simulation. Duplicate
    90-item pattern audits require the full response vector or administrative
    batch review and are not directly observable before a prospective
    15-item-first routing decision.
    """
    rows = []
    for sample, g in [
        ("Overall", d),
        ("mapped Hunan code group", d[d.region == "Hunan"]),
        ("mapped Shandong code group", d[d.region == "Shandong"]),
    ]:
        g = g.copy().reset_index(drop=True)
        quality_flag = (g["all_same"] | g["all_zero"] | g["duplicate_item_pattern"]).to_numpy()
        eligible = g.loc[~quality_flag].copy().reset_index(drop=True)
        row = {
            "sample": sample,
            "n": int(len(g)),
            "quality_audit_or_retest_n": int(quality_flag.sum()),
            "quality_audit_or_retest_proportion": float(quality_flag.mean()) if len(g) else np.nan,
            "eligible_for_15_item_workflow_n": int(len(eligible)),
            "all_same_n": int(g["all_same"].sum()),
            "all_zero_n": int(g["all_zero"].sum()),
            "duplicate_90_item_pattern_n": int(g["duplicate_item_pattern"].sum()),
            "response_pattern_flag_overlap_note": "Response-pattern flags are not mutually exclusive. Audit/review counts refer to the union of available flagged records rather than the sum of individual flag counts.",
            "audit_rule": "All-same, all-zero, or duplicated 90-item response patterns are routed to quality audit/retest/full-form review in a retrospective full-form audit simulation.",
        }
        if len(eligible) == 0 or eligible.label_ge160.nunique() < 2:
            row["note"] = "insufficient eligible records after response-pattern audit"
            rows.append(row)
            continue
        m = selective_escalation_metrics(eligible, items, low_cut, high_cut, sample + " after response-pattern audit")
        middle_or_quality_review = m["middle_escalation_n"] + int(quality_flag.sum())
        accepted_total = m["accepted_n"]
        avg_items = (15 * accepted_total + 90 * middle_or_quality_review) / len(g)
        row.update({
            "low_cut_rawsum_le": low_cut,
            "high_cut_rawsum_ge": high_cut,
            "provisional_low_zone_n": m["low_zone_n"],
            "automatic_full_form_or_quality_review_n": int(middle_or_quality_review),
            "priority_human_review_n": m["high_zone_n"],
            "accepted_15_item_route_proportion": float(accepted_total / len(g)),
            "full_form_quality_review_or_retest_proportion": float(middle_or_quality_review / len(g)),
            "eligible_low_zone_npv": m["low_zone_npv"],
            "eligible_high_zone_ppv": m["high_zone_ppv"],
            "eligible_accepted_error_rate_all_eligible": m["accepted_error_rate_all"],
            "whole_sample_accepted_error_rate": float(m["accepted_error_n"] / len(g)),
            "average_item_burden_if_flagged_records_receive_full_review": float(avg_items),
            "burden_reduction_vs_90": float((90 - avg_items) / 90),
            "deployment_caution": "This is a retrospective response-quality audit simulation based on available full-form response-pattern information. Duplicate 90-item pattern audits are retrospective/administrative checks and are not directly observable before a prospective 15-item-first routing decision; local recalibration remains required.",
        })
        rows.append(row)
    return pd.DataFrame(rows)


def first_stage_observable_quality_flags(d: pd.DataFrame, items: Sequence[str], low_cut: int = 13, high_cut: int = 24) -> pd.DataFrame:
    """Evaluate quality flags observable from the fixed 15-item first stage."""
    rows = []
    item_cols = list(items)
    for sample, g in [
        ("Overall", d),
        ("mapped Hunan code group", d[d.region == "Hunan"]),
        ("mapped Shandong code group", d[d.region == "Shandong"]),
    ]:
        g = g.copy().reset_index(drop=True)
        if g.empty:
            continue
        first_stage = g[item_cols]
        flag_all_same_15 = first_stage.nunique(axis=1).eq(1).to_numpy()
        flag_all_zero_15 = first_stage.sum(axis=1).eq(0).to_numpy()
        flag_duplicate_15 = first_stage.duplicated(keep=False).to_numpy()
        flag_any_15 = flag_all_same_15 | flag_all_zero_15 | flag_duplicate_15
        eligible = g.loc[~flag_any_15].copy().reset_index(drop=True)
        row = {
            "sample": sample,
            "n": int(len(g)),
            "first_stage_all_same_15_n": int(flag_all_same_15.sum()),
            "first_stage_all_same_15_percent": float(flag_all_same_15.mean()),
            "first_stage_all_zero_15_n": int(flag_all_zero_15.sum()),
            "first_stage_all_zero_15_percent": float(flag_all_zero_15.mean()),
            "first_stage_duplicate_15_pattern_n": int(flag_duplicate_15.sum()),
            "first_stage_duplicate_15_pattern_percent": float(flag_duplicate_15.mean()),
            "first_stage_any_observable_flag_n": int(flag_any_15.sum()),
            "first_stage_any_observable_flag_percent": float(flag_any_15.mean()),
            "non_flagged_first_stage_n": int(len(eligible)),
            "interpretation": "These flags are observable from the fixed 15-item first stage. Duplicate 15-item patterns are expected to be less specific than full-form duplicate patterns and should be used only for batch-level monitoring, not individual exclusion.",
        }
        if len(eligible) > 0 and eligible.label_ge160.nunique() >= 2:
            m = selective_escalation_metrics(eligible, items, low_cut, high_cut, sample + " without first-stage observable flags")
            row.update({
                "non_flagged_low_zone_npv": m["low_zone_npv"],
                "non_flagged_high_zone_ppv": m["high_zone_ppv"],
                "non_flagged_accepted_error_rate_all": m["accepted_error_rate_all"],
                "non_flagged_escalation_proportion": m["escalation_proportion"],
                "non_flagged_average_item_burden": m["average_item_burden"],
                "non_flagged_burden_reduction_vs_90": m["burden_reduction_vs_90"],
                "non_flagged_false_negative_low_zone_n": m["false_negative_low_zone_n"],
                "non_flagged_false_positive_high_zone_n": m["false_positive_high_zone_n"],
            })
        else:
            row["note"] = "insufficient non-flagged first-stage records for workflow metrics"
        rows.append(row)
    return pd.DataFrame(rows)


def nonoverlap_and_severity_validation(shandong: pd.DataFrame, selected: Sequence[str], p_score: np.ndarray) -> pd.DataFrame:
    rem_items = [i for i in ITEMS if i not in selected]
    labels = [
        ("Full 90-item total score >=160", "Primary parent total-score screening rule", shandong.label_ge160.to_numpy()),
        ("Remaining 75-item raw sum >=59", "Non-overlapping remaining-item validation after excluding the 15 operational items", (shandong[rem_items].sum(axis=1) >= 59).astype(int).to_numpy()),
        ("Top decile of full 90-item raw sum", "High symptom-burden prioritization label within the mapped Shandong validation group", (shandong.item_sum_0_4 >= shandong.item_sum_0_4.quantile(.90, interpolation="lower")).astype(int).to_numpy()),
        ("SCL-90 total score >=200", "Higher-severity total-score threshold sensitivity analysis", (shandong.total >= 200).astype(int).to_numpy()),
    ]
    rows = []
    for label, definition, y in labels:
        ev = evaluate(y, p_score, None)
        rows.append({
            "label": label,
            "definition": definition,
            "positive_n": int(y.sum()),
            "positive_percent": 100 * float(y.mean()),
            "roc_auc": ev["roc_auc"],
            "pr_auc": ev["pr_auc"],
            "top10_recall_percent": 100 * ev["top10_recall"],
            "top10_screen_positive_rate_percent": 100 * ev["top10_precision"],
            "top20_recall_percent": 100 * ev["top20_recall"],
            "interpretive_note": "Ranking validation only; labels are still derived from SCL-90 item responses, not independent clinical outcomes.",
        })
    return pd.DataFrame(rows)


def subgroup_workflow_diagnostics(hunan: pd.DataFrame, shandong: pd.DataFrame, selected: Sequence[str], p_h: np.ndarray, p_s: np.ndarray, model_threshold: float, low_cut: int, high_cut: int) -> pd.DataFrame:
    """Bias/transportability diagnostics only; subgroup codebooks are not independently verified."""
    frames = [
        ("mapped Hunan code group", hunan, p_h),
        ("mapped Shandong code group", shandong, p_s),
    ]
    rows = []
    for label, df, p in frames:
        group_specs = [("location_code_group", label, np.ones(len(df), dtype=bool))]
        for code in sorted(df.gender.dropna().unique()):
            group_specs.append(("gender_code", f"{int(code)} within {label}", df.gender.eq(code).to_numpy()))
        quality_flag = (df["all_same"] | df["duplicate_item_pattern"]).to_numpy()
        group_specs.extend([
            ("available_response_pattern_check", f"flagged within {label}", quality_flag),
            ("available_response_pattern_check", f"not flagged within {label}", ~quality_flag),
        ])
        for group_type, group, mask in group_specs:
            sub = df.loc[mask].reset_index(drop=True)
            pp = p[mask]
            if len(sub) == 0 or sub.label_ge160.nunique() < 2:
                continue
            ev = evaluate(sub.label_ge160.to_numpy(), pp, model_threshold)
            wf = selective_escalation_metrics(sub, selected, low_cut, high_cut, group)
            rows.append({
                "group_type": group_type,
                "group": group,
                "n": int(len(sub)),
                "positive_n": int(sub.label_ge160.sum()),
                "prevalence": float(sub.label_ge160.mean()),
                "roc_auc": ev["roc_auc"],
                "pr_auc": ev["pr_auc"],
                "low_zone_npv": wf["low_zone_npv"],
                "high_zone_ppv": wf["high_zone_ppv"],
                "accepted_error_rate_all": wf["accepted_error_rate_all"],
                "escalation_proportion": wf["escalation_proportion"],
                "average_item_burden": wf["average_item_burden"],
                "interpretive_note": "Descriptive subgroup check only; not for clinical diagnosis. Gender/location code meanings were not independently codebook-verified.",
            })
    return pd.DataFrame(rows)


def tripod_ai_reporting_checklist() -> pd.DataFrame:
    rows = [
        ("Title/abstract", "Identify prediction-like workflow and intended target", "Addressed in title/abstract as full-score rule approximation and triage workflow, not diagnosis."),
        ("Background", "Explain clinical/service context and intended use", "Framed as service-capacity triage for an existing SCL-90 total-score rule."),
        ("Data source", "Describe source, eligibility, and availability", "Public Mendeley Data V1, DOI and license reported; raw item wording not redistributed."),
        ("Outcome", "Define target outcome and timing", "Full-length SCL-90 total >=160; same item pool limitation explicitly stated."),
        ("Predictors", "Describe candidate predictors and preprocessing", "S1-S90 numeric items; fixed archived 15-item operational example and comparators."),
        ("Sample split", "Describe development/validation approach", "Metadata-supported mapped Hunan/Shandong location-code split plus reverse and CV checks."),
        ("Model", "Specify algorithms and hyperparameters", "Logistic models, ranking approaches, tree-based sensitivity, and raw-sum workflow documented in code."),
        ("Performance", "Report discrimination, calibration/ranking, and uncertainty", "ROC-AUC, PR-AUC, Brier/calibration, top-k capacity, bootstrap CIs, and nested checks."),
        ("Fairness/heterogeneity", "Evaluate subgroup or transportability diagnostics", "Gender-code/location/quality-flag diagnostics reported without overinterpreting code meanings."),
        ("Deployment", "State intended and non-intended uses", "Deployment card specifies human review, response-quality safeguard, calibration, and crisis override requirements."),
        ("Transparency", "Provide code, environment, and reproducibility route", "S1 Code includes requirements, environment report, smoke tests, quick route, and full route."),
    ]
    return pd.DataFrame(rows, columns=["reporting_domain", "transparency_item", "manuscript_or_package_response"])


def deployment_card() -> pd.DataFrame:
    rows = [
        ("Intended use", "First-stage low-burden approximation of an existing full-form SCL-90 total-score rule for service-capacity triage."),
        ("Not intended for", "Clinical diagnosis, crisis detection, self-harm risk prediction, treatment-need determination, or automated denial/exclusion from support."),
        ("Minimum safeguards", "Human review, automatic full-form escalation for uncertain cases, response-quality review pathway, local calibration/audit, subgroup monitoring, and data-security controls."),
        ("Required human override", "Crisis signal, direct help-seeking, counselor/staff concern, prior service contact, or local safeguarding policy overrides model rank at any stage."),
        ("Known failure modes", "All-zero/all-same responses, duplicated patterns, unverified location/gender code meanings, distribution shift, no response-time or attention-check metadata."),
        ("Probability interpretation", "Weighted logistic scores are ranking scores; individual probability thresholds require local recalibration before operational use."),
        ("Order-assumed domain-balanced caveat", "Order-assumed domain-balanced analysis assumes conventional SCL-90 item order because item wording was not redistributed in the public dataset."),
        ("Evidence boundary", "No independent clinical criterion, diagnosis, crisis outcome, help-seeking outcome, or longitudinal service outcome was evaluated."),
    ]
    return pd.DataFrame(rows, columns=["card_item", "statement"])


def fully_nested_region_stratified_cv(d: pd.DataFrame, methods: Sequence[str] = ("Correlation ranking", "Mutual information ranking", "LASSO/L1 ranking")) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Repeat item selection inside each region-stratified CV fold."""
    y = d.label_ge160.to_numpy()
    strat = np.array([f"{yy}_{rr}" for yy, rr in zip(y, d.region.astype(str).to_numpy())])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    item_rows = []
    for method in methods:
        for fold, (tr, te) in enumerate(skf.split(d[ITEMS], strat), 1):
            train = d.iloc[tr].reset_index(drop=True)
            test = d.iloc[te].reset_index(drop=True)
            if method == "Correlation ranking":
                ranking = rank_items_correlation(train[ITEMS], train.label_ge160.to_numpy())
            elif method == "Mutual information ranking":
                ranking = rank_items_mi(train[ITEMS], train.label_ge160.to_numpy())
            elif method == "LASSO/L1 ranking":
                ranking = rank_items_lasso_algorithmic(train[ITEMS], train.label_ge160.to_numpy())
            else:
                raise ValueError(f"Unsupported nested CV method: {method}")
            items = ranking.item.head(15).tolist()
            ev, _, _, _, _, _ = fit_eval_itemset(train, test, items, weighted=True)
            rows.append({"method": method, "fold": fold, "selected_items": ",".join(items), **ev})
            for item in items:
                item_rows.append({"method": method, "fold": fold, "item": item, "domain": ITEM_TO_DOMAIN.get(item, "Unmapped")})
    folds = pd.DataFrame(rows)
    summary = folds.drop(columns=["fold", "selected_items"]).groupby("method").agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join([str(c) for c in col if c != ""]).strip("_") for col in summary.columns.to_flat_index()]
    stability = pd.DataFrame(item_rows).groupby(["method", "item", "domain"]).size().reset_index(name="fold_selection_count").sort_values(["method", "fold_selection_count", "item"], ascending=[True, False, True])
    return folds, summary, stability


def nested_selective_escalation_cv(d: pd.DataFrame, methods: Sequence[str] = ("Correlation ranking", "Mutual information ranking", "LASSO/L1 ranking")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat item selection and selective-escalation threshold selection inside each CV fold."""
    y = d.label_ge160.to_numpy()
    strat = np.array([f"{yy}_{rr}" for yy, rr in zip(y, d.region.astype(str).to_numpy())])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for method in methods:
        for fold, (tr, te) in enumerate(skf.split(d[ITEMS], strat), 1):
            train = d.iloc[tr].reset_index(drop=True)
            test = d.iloc[te].reset_index(drop=True)
            if method == "Correlation ranking":
                ranking = rank_items_correlation(train[ITEMS], train.label_ge160.to_numpy())
            elif method == "Mutual information ranking":
                ranking = rank_items_mi(train[ITEMS], train.label_ge160.to_numpy())
            elif method == "LASSO/L1 ranking":
                ranking = rank_items_lasso_algorithmic(train[ITEMS], train.label_ge160.to_numpy())
            else:
                raise ValueError(f"Unsupported nested workflow CV method: {method}")
            items = ranking.item.head(15).tolist()
            grid = selective_escalation_threshold_grid(train, items)
            if grid.empty:
                rows.append({
                    "method": method,
                    "fold": fold,
                    "selected_items": ",".join(items),
                    "note": "no threshold pair met the prespecified development constraints in this fold",
                })
                continue
            low_cut = int(grid.iloc[0]["low_cut_rawsum_le"])
            high_cut = int(grid.iloc[0]["high_cut_rawsum_ge"])
            dev = selective_escalation_metrics(train, items, low_cut, high_cut, "nested training fold")
            ev = selective_escalation_metrics(test, items, low_cut, high_cut, "nested test fold")
            rows.append({
                "method": method,
                "fold": fold,
                "selected_items": ",".join(items),
                "low_cut_rawsum_le": low_cut,
                "high_cut_rawsum_ge": high_cut,
                "development_low_zone_npv": dev["low_zone_npv"],
                "development_high_zone_ppv": dev["high_zone_ppv"],
                "development_accepted_error_rate_all": dev["accepted_error_rate_all"],
                "test_n": ev["n"],
                "test_positive_n": ev["positive_n"],
                "test_low_zone_npv": ev["low_zone_npv"],
                "test_high_zone_ppv": ev["high_zone_ppv"],
                "test_accepted_error_rate_all": ev["accepted_error_rate_all"],
                "test_accepted_error_rate_accepted": ev["accepted_error_rate_accepted"],
                "test_escalation_proportion": ev["escalation_proportion"],
                "test_average_item_burden": ev["average_item_burden"],
                "test_burden_reduction_vs_90": ev["burden_reduction_vs_90"],
                "test_false_negative_low_zone_n": ev["false_negative_low_zone_n"],
                "test_false_positive_high_zone_n": ev["false_positive_high_zone_n"],
                "workflow_note": "Nested CV repeats item ranking, 15-item selection, and low/high raw-sum threshold selection inside each training fold.",
            })
    folds = pd.DataFrame(rows)
    metric_cols = [
        "low_cut_rawsum_le",
        "high_cut_rawsum_ge",
        "development_low_zone_npv",
        "development_high_zone_ppv",
        "development_accepted_error_rate_all",
        "test_low_zone_npv",
        "test_high_zone_ppv",
        "test_accepted_error_rate_all",
        "test_accepted_error_rate_accepted",
        "test_escalation_proportion",
        "test_average_item_burden",
        "test_burden_reduction_vs_90",
        "test_false_negative_low_zone_n",
        "test_false_positive_high_zone_n",
    ]
    present = [c for c in metric_cols if c in folds.columns]
    summary = folds.groupby("method")[present].agg(["mean", "std", "min", "max"]).reset_index()
    summary.columns = ["_".join([str(c) for c in col if c != ""]).strip("_") for col in summary.columns.to_flat_index()]
    return folds, summary


def response_pattern_audit_table(d: pd.DataFrame) -> pd.DataFrame:
    patterns = [
        ("all-same response pattern", "all_same", "Retained in primary analysis", "Excluded in response-quality sensitivity", "Quality review or retest before provisionally low-burden routing"),
        ("all-zero response pattern", "all_zero", "Retained in primary analysis", "Captured within all-same sensitivity", "Flag as low-symptom but quality-sensitive; do not assume validity without local QA"),
        ("duplicate 90-item pattern", "duplicate_item_pattern", "Retained in primary analysis", "Deduplicated by retaining first occurrence", "Administrative audit; avoid treating duplicated patterns as independent evidence"),
        ("not flagged by available response-pattern checks", None, "Entered into 15-item triage or selective escalation", "Reference group", "Proceed through local triage workflow with human-review safeguards"),
    ]
    rows = []
    for pattern, col, primary, sensitivity, recommendation in patterns:
        for sample, g in [("Overall", d), ("mapped Hunan code group", d[d.region == "Hunan"]), ("mapped Shandong code group", d[d.region == "Shandong"] )]:
            if col is None:
                mask = ~(g["all_same"] | g["all_zero"] | g["duplicate_item_pattern"])
            else:
                mask = g[col]
            rows.append({
                "pattern": pattern,
                "sample": sample,
                "n": int(mask.sum()),
                "percent": 100 * float(mask.mean()),
                "primary_analysis_handling": primary,
                "sensitivity_analysis_handling": sensitivity,
                "deployment_recommendation": recommendation,
            })
    return pd.DataFrame(rows)


def write_environment(out: Path, quick: bool = False, skip_nested_cv: bool = False, bootstrap_ci: int = 500) -> None:
    import importlib.metadata as md
    pkgs = ["numpy", "pandas", "scikit-learn", "matplotlib", "xgboost", "lightgbm"]
    versions = {}
    for p in pkgs:
        try: versions[p] = md.version(p)
        except Exception: versions[p] = None
    with open(out/"environment_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "python": sys.version,
            "platform": platform.platform(),
            "packages": versions,
            "package_version": "1.0.0",
            "report_scope": "Runtime environment observed for this run; this report is not a dependency lock file.",
            "run_mode": "quick" if quick else ("custom" if skip_nested_cv or bootstrap_ci != 500 else "full"),
            "tree_based_sensitivity_run": False,
        }, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs_run"))
    parser.add_argument("--random-repeats", type=int, default=1000)
    parser.add_argument("--bootstrap-ci", type=int, default=500)
    parser.add_argument("--quick", action="store_true", help="Reviewer smoke-run mode: set bootstrap resampling to 0 and keep archived simulations unless explicitly recomputing.")
    parser.add_argument("--skip-nested-cv", action="store_true", help="Skip fully nested CV and nested workflow CV for a faster reviewer smoke-run.")
    parser.add_argument("--write-algorithmic-lasso-ranking", action="store_true", help="Also export the raw current-environment LASSO/L1 item ranking. A current-environment sensitivity table is always generated.")
    parser.add_argument("--recompute-random-baseline", action="store_true", help="Recompute the 1000x5 random raw-score item-set distributions instead of using archived aggregate draws")
    parser.add_argument("--write-individual-level-outputs", action="store_true", help="Write excluded raw record and individual-level Shandong predictions. Off by default for sharing.")
    args = parser.parse_args()
    if args.quick:
        args.bootstrap_ci = 0
        args.skip_nested_cv = True
        if args.recompute_random_baseline:
            args.random_repeats = min(args.random_repeats, 5)
    tables, _ = ensure_dirs(args.out)
    static_dir = Path(__file__).resolve().parent / "static"
    log_step("Loading public CSV and applying schema / row-level quality checks")
    d, excluded = load_and_qc(args.data)
    lasso_rank = read_archived_lasso(static_dir)
    main_items = lasso_rank.item.head(15).tolist()
    if args.write_individual_level_outputs:
        excluded.to_csv(tables/"excluded_invalid_record.csv", index=False)
    sample_table(d).to_csv(tables/"table1_sample_characteristics_quality.csv", index=False)
    outcome_table(d, main_items).to_csv(tables/"table2_outcome_definitions_prevalence.csv", index=False)
    hunan = d[d.region == "Hunan"].copy().reset_index(drop=True)
    shandong = d[d.region == "Shandong"].copy().reset_index(drop=True)

    log_step("Ranking items and evaluating fixed/candidate item sets")
    rankings = {
        "LASSO/L1 ranking": lasso_rank,
        "Correlation ranking": rank_items_correlation(hunan[ITEMS], hunan.label_ge160.to_numpy()),
        "Mutual information ranking": rank_items_mi(hunan[ITEMS], hunan.label_ge160.to_numpy()),
        "Elastic net ranking (SGD)": rank_items_elastic_net_sgd(hunan[ITEMS], hunan.label_ge160.to_numpy()),
    }
    for name, r in rankings.items():
        r.to_csv(tables/safe_filename(name), index=False)
    algorithmic_lasso_rank = rank_items_lasso_algorithmic(hunan[ITEMS], hunan.label_ge160.to_numpy())
    domain_balanced_15, domain_balanced_selection_15 = domain_balanced_items(hunan, rankings["Correlation ranking"], n_items=15)
    domain_balanced_20, domain_balanced_selection_20 = domain_balanced_items(hunan, rankings["Correlation ranking"], n_items=20)
    pd.concat([
        domain_balanced_selection_15.assign(item_count=15),
        domain_balanced_selection_20.assign(item_count=20),
    ], ignore_index=True).to_csv(tables/"tableS_order_assumed_domain_balanced_item_selection.csv", index=False)
    if args.write_algorithmic_lasso_ranking:
        algorithmic_lasso_rank.to_csv(tables/"lasso_l1_algorithmic_item_ranking_current_environment.csv", index=False)

    rows = []; models = {}
    for name, ranking in rankings.items():
        for k in [5,10,15,20,30]:
            selected = ranking.item.head(k).tolist()
            ev, model, scaler, thr, p_tr, p_te = fit_eval_itemset(hunan, shandong, selected, weighted=True)
            rows.append({"analysis": name, "method": name, "item_count": k, "selected_items": ",".join(selected), **ev})
            if name == "LASSO/L1 ranking" and k == 15:
                models["main"] = (selected, model, scaler, thr, p_tr, p_te)
    for k, db_items in [(15, domain_balanced_15), (20, domain_balanced_20)]:
        ev, model, scaler, db_thr, db_p_tr, db_p_te = fit_eval_itemset(hunan, shandong, db_items, weighted=True)
        rows.append({"analysis": f"Order-assumed domain-balanced {k}-item comparator with correlation-ranked fill", "method": "Order-assumed domain-balanced comparator", "item_count": k, "selected_items": ",".join(db_items), **ev})

    # Raw-sum comparator for the fixed 15-item operational example. This is a
    # redundancy/simple-score benchmark, not a calibrated probability model.
    raw15 = shandong[main_items].sum(axis=1).to_numpy()
    raw15_ev = evaluate(shandong.label_ge160.to_numpy(), raw15, None)
    rows.append({"analysis":"Fixed 15-item operational raw-sum score", "method":"Raw sum benchmark", "item_count":15, "selected_items": ",".join(main_items), **raw15_ev})
    ev90, _, _, _, _, _ = fit_eval_itemset(hunan, shandong, ITEMS, weighted=True)
    rows.append({"analysis":"Full 90 upper bound", "method":"Full 90 upper bound", "item_count":90, "selected_items": ",".join(ITEMS), **ev90})
    pd.DataFrame(rows).to_csv(tables/"table3_performance_by_item_count_and_selection.csv", index=False)

    log_step("Writing fixed 15-item model parameters, bootstrap CIs, and top-k triage tables")
    selected, model, scaler, thr, p_h, p_s = models["main"]
    std_coef = model.coef_[0]; raw_coef = std_coef / scaler.scale_; intercept_raw = model.intercept_[0] - np.sum(std_coef * scaler.mean_ / scaler.scale_)
    stability_path = static_dir/"lasso_stability_1000.csv"
    stability = pd.read_csv(stability_path)
    stability.to_csv(tables/"lasso_bootstrap_stability_1000.csv", index=False)
    par = pd.DataFrame({"item": selected, "development_mean": scaler.mean_, "development_sd": scaler.scale_, "standardized_beta": std_coef, "raw_score_beta": raw_coef})
    par = par.merge(stability, on="item", how="left")
    par = pd.concat([par, pd.DataFrame([{"item":"Intercept", "standardized_beta": model.intercept_[0], "raw_score_beta": intercept_raw}])], ignore_index=True)
    par.to_csv(tables/"table4_main_15_item_model_parameters.csv", index=False)
    if args.write_individual_level_outputs:
        pred = shandong[["location","region","gender","total","item_sum_0_4","label_ge160","all_same","all_zero"]].copy()
        pred["predicted_probability_weighted_lasso15"] = p_s
        pred["main_lasso15_raw_sum"] = shandong[selected].sum(axis=1)
        pred.to_csv(tables/"shandong_external_predictions_main_lasso15.csv", index=False)
    main_ci = bootstrap_ci(shandong.label_ge160.to_numpy(), p_s, thr, args.bootstrap_ci)
    main_ci.to_csv(tables/f"main_hunan_to_shandong_bootstrap_ci_{args.bootstrap_ci}.csv", index=False)
    topk_bootstrap_ci_table(evaluate(shandong.label_ge160.to_numpy(), p_s, thr), main_ci, n_validation=len(shandong)).to_csv(tables/"tableS_topk_triage_bootstrap_ci.csv", index=False)

    # Current-environment LASSO sensitivity: demonstrates that the archived
    # operational item order is not a cherry-picked sequence. This table is
    # regenerated from the active software environment.
    current_items = algorithmic_lasso_rank.item.head(15).tolist()
    current_ev, _, _, current_thr, _, current_p = fit_eval_itemset(hunan, shandong, current_items, weighted=True)
    pd.DataFrame([
        {"ranking_source":"Archived operational LASSO/L1 ranking", "selected_items": ",".join(selected), **evaluate(shandong.label_ge160.to_numpy(), p_s, thr)},
        {"ranking_source":"Current-environment algorithmic LASSO/L1 rerun", "selected_items": ",".join(current_items), **current_ev},
    ]).to_csv(tables/"tableS_current_environment_lasso_sensitivity.csv", index=False)

    log_step("Preparing random raw-score baseline outputs")
    if args.recompute_random_baseline:
        dist, summ = random_rawsum_baseline(shandong, [5,10,15,20,30], args.random_repeats, RANDOM_STATE)
        random_baseline_source = f"recomputed_{args.random_repeats}_each_size"
        dist.to_csv(tables/"random_rawsum_item_set_distributions.csv", index=False)
        summ.to_csv(tables/"table5_random_rawsum_item_set_summary.csv", index=False)
    else:
        if args.random_repeats != 1000:
            warnings.warn(
                "--random-repeats is ignored unless --recompute-random-baseline is supplied; "
                "using archived 1000-each-size random-baseline outputs.",
                RuntimeWarning,
            )
        dist_static = static_dir/"random_sets_1000.csv"
        summ_static = static_dir/"random_summary_1000.csv"
        dist = pd.read_csv(dist_static)
        summ = pd.read_csv(summ_static)
        random_baseline_source = "archived_1000_each_size"
        # Preserve the archived values byte-for-byte in the canonical output files.
        shutil.copy2(dist_static, tables/"random_rawsum_item_set_distributions.csv")
        shutil.copy2(summ_static, tables/"table5_random_rawsum_item_set_summary.csv")

    # Reverse location-code validation. Use the archived Shandong LASSO sensitivity ranking when available
    # to keep the manuscript table stable across solver versions.
    log_step("Running reverse mapped-location validation and calibration summaries")
    rev_static = static_dir/"lasso_rank_shandong.csv"
    if rev_static.exists():
        rev_rank = pd.read_csv(rev_static)
        if "importance" not in rev_rank.columns:
            if "abs_score" in rev_rank.columns:
                rev_rank = rev_rank.rename(columns={"abs_score":"importance"})
            elif "score" in rev_rank.columns:
                rev_rank["importance"] = rev_rank["score"].abs()
        rev_rank = rev_rank[["item", "importance"]].sort_values("importance", ascending=False).reset_index(drop=True)
        rev_rank.to_csv(tables/"reverse_shandong_lasso_item_ranking_all90.csv", index=False)
    else:
        rev_rank = rank_items_correlation(shandong[ITEMS], shandong.label_ge160.to_numpy())
        rev_rank.to_csv(tables/"reverse_shandong_correlation_item_ranking_all90.csv", index=False)
    rev_items = rev_rank.item.head(15).tolist()
    ev_rev, _, _, rev_thr, _, p_rev = fit_eval_itemset(shandong, hunan, rev_items, weighted=True)
    transport = pd.DataFrame([
        {"analysis":"Hunan development -> Shandong validation, LASSO 15", "direction":"Hunan->Shandong", "selected_items": ",".join(selected), **evaluate(shandong.label_ge160.to_numpy(), p_s, thr)},
        {"analysis":"Shandong development -> Hunan validation, LASSO 15", "direction":"Shandong->Hunan", "selected_items": ",".join(rev_items), **ev_rev},
    ])
    transport.to_csv(tables/"table6_regional_transportability.csv", index=False)

    # Target-region split recalibration and calibration curve bins for Fig 4.
    cal_idx, test_idx = train_test_split(np.arange(len(shandong)), test_size=0.5, random_state=RANDOM_STATE, stratify=shandong.label_ge160)
    y_all = shandong.label_ge160.to_numpy(); p_all = p_s
    eps=1e-6; lp = np.log(np.clip(p_all, eps, 1-eps) / np.clip(1-p_all, eps, 1-eps))
    y_test, p_test, lp_test = y_all[test_idx], p_all[test_idx], lp[test_idx]
    rec = [{"analysis":"Hunan->Shandong: weighted original", "calibration_subset_n": len(cal_idx), "test_subset_n": len(test_idx), **evaluate(y_test, p_test, thr)}]
    # intercept-only recalibration: fit intercept with fixed slope 1 by fitting offset approximately through 1D intercept search
    def sigmoid(z): return 1/(1+np.exp(-z))
    alphas = np.linspace(-4,4,2001)
    losses = [brier_score_loss(y_all[cal_idx], sigmoid(lp[cal_idx]+a)) for a in alphas]
    alpha = float(alphas[int(np.argmin(losses))])
    p_int = sigmoid(lp_test + alpha)
    tmp = evaluate(y_test, p_int, None); tmp.update({"analysis":"Hunan->Shandong: weighted intercept-only recalibrated", "calibration_subset_n": len(cal_idx), "test_subset_n": len(test_idx), "recalibration_alpha": alpha, "recalibration_beta": np.nan}); rec.append(tmp)
    lr = LogisticRegression(C=1e6, solver="liblinear", max_iter=200).fit(lp[cal_idx].reshape(-1,1), y_all[cal_idx])
    p_recal = lr.predict_proba(lp_test.reshape(-1,1))[:,1]
    tmp = evaluate(y_test, p_recal, None); tmp.update({"analysis":"Hunan->Shandong: weighted slope+intercept recalibrated", "calibration_subset_n": len(cal_idx), "test_subset_n": len(test_idx), "recalibration_alpha": float(lr.intercept_[0]), "recalibration_beta": float(lr.coef_[0,0])}); rec.append(tmp)
    pd.DataFrame(rec).to_csv(tables/"table7_calibration_and_recalibration_target_region_split.csv", index=False)
    bins = pd.concat([calibration_bins(y_test, p_test, "Original weighted model"), calibration_bins(y_test, p_recal, "Slope+intercept recalibrated model")], ignore_index=True)
    bins.to_csv(tables/"calibration_curve_bins_figure5.csv", index=False)

    # Sensitivity labels.
    log_step("Running label, non-overlap, and response-quality sensitivity analyses")
    sens_rows = []
    ev, _, _, _, _, _ = fit_eval_itemset(hunan.assign(label=hunan.label_gt160), shandong.assign(label=shandong.label_gt160), selected, label="label", weighted=True)
    sens_rows.append({"analysis":"Alternative cutoff: total score >160", **ev})
    rem_items = [i for i in ITEMS if i not in selected]; rem_thr = 59
    y_rem = (shandong[rem_items].sum(axis=1) >= rem_thr).astype(int).to_numpy()
    sens_rows.append({"analysis":"Remaining 75-item label >=59", "remaining_item_threshold":"75-item raw sum >=59", **evaluate(y_rem, p_s, thr)})
    raw_score = shandong[selected].sum(axis=1).to_numpy()
    sens_rows.append({"analysis":"Unweighted raw sum of selected 15 items", **evaluate(shandong.label_ge160.to_numpy(), raw_score, None)})
    pd.DataFrame(sens_rows).to_csv(tables/"table8_label_and_data_quality_sensitivity.csv", index=False)
    data_quality_exclusion_sensitivity(d, selected).to_csv(tables/"tableS_data_quality_exclusion_sensitivity.csv", index=False)

    # Subgroups.
    log_step("Writing subgroup and capacity-constrained triage summaries")
    sg_rows=[]
    for code in [1,2]:
        m = shandong.gender.eq(code).to_numpy()
        sg_rows.append({"analysis": f"Shandong validation gender code {code}", "region":"Shandong", "gender_code": code, **evaluate(y_all[m], p_all[m], thr)})
    sg_rows.append({"analysis":"Region-level: Shandong under Hunan-developed model", "region":"Shandong", "gender_code":"all", **evaluate(y_all, p_all, thr)})
    pd.DataFrame(sg_rows).to_csv(tables/"table9_subgroup_performance_and_calibration.csv", index=False)

    # Triage capacity.
    tm = topk_metrics(y_all, p_all, [5,10,15,20,25,30]); base = y_all.mean(); triage=[]
    for k in [5,10,15,20,25,30]:
        theoretical_max = min(1.0, (k / 100) / base) if base else np.nan
        observed = tm[f"top{k}_recall"]
        triage.append({"capacity_percent": k, "students_to_review_in_shandong": int(round(len(y_all)*k/100)), "coverage_recall_of_full_score_screen_positive": observed, "theoretical_max_recall_given_capacity": theoretical_max, "percent_of_theoretical_max": observed / theoretical_max if theoretical_max else np.nan, "positive_rate_in_reviewed_group": tm[f"top{k}_precision"], "lift_over_base_rate": tm[f"top{k}_precision"]/base})
    pd.DataFrame(triage).to_csv(tables/"table10_triage_capacity_coverage.csv", index=False)

    # Selective full-form escalation using development-set threshold selection and fixed hold-out evaluation.
    log_step("Selecting and evaluating selective full-form escalation thresholds")
    threshold_grid = selective_escalation_threshold_grid(hunan, selected)
    threshold_grid.to_csv(tables/"tableS_selective_escalation_threshold_grid_hunan.csv", index=False)
    if threshold_grid.empty:
        raise RuntimeError("No development threshold pair meets all prespecified constraints; do not apply an unvalidated fallback pair.")
    low_cut, high_cut = int(threshold_grid.iloc[0]["low_cut_rawsum_le"]), int(threshold_grid.iloc[0]["high_cut_rawsum_ge"])
    selective_rows = [
        selective_escalation_metrics(hunan, selected, low_cut, high_cut, "mapped Hunan development group"),
        selective_escalation_metrics(shandong, selected, low_cut, high_cut, "mapped Shandong hold-out group"),
    ]
    pd.DataFrame(selective_rows).to_csv(tables/"table11_selective_full_form_escalation.csv", index=False)
    selective_escalation_bootstrap_ci(shandong, selected, low_cut, high_cut, args.bootstrap_ci).to_csv(tables/"tableS_selective_escalation_bootstrap_ci.csv", index=False)
    selective_escalation_error_profile(shandong, selected, low_cut, high_cut, "mapped Shandong hold-out group").to_csv(tables/"tableS_selective_escalation_error_profile.csv", index=False)
    response_quality_restricted_selective_escalation_sensitivity(d, selected).to_csv(tables/"tableS_response_quality_restricted_selective_escalation.csv", index=False)
    selective_escalation_threshold_tradeoff(hunan, shandong, selected, low_cut, high_cut).to_csv(tables/"tableS_selective_escalation_threshold_tradeoff.csv", index=False)
    retrospective_response_quality_audit_simulation(d, selected, low_cut, high_cut).to_csv(tables/"tableS_retrospective_response_quality_audit.csv", index=False)
    first_stage_observable_quality_flags(d, selected, low_cut, high_cut).to_csv(tables/"tableS_first_stage_observable_quality_flags.csv", index=False)
    subgroup_workflow_diagnostics(hunan, shandong, selected, p_h, p_s, thr, low_cut, high_cut).to_csv(tables/"tableS_subgroup_workflow_diagnostics.csv", index=False)

    nonoverlap_and_severity_validation(shandong, selected, p_s).to_csv(tables/"table12_nonoverlap_and_severity_validation.csv", index=False)

    if args.skip_nested_cv:
        log_step("Skipping nested CV outputs by request; use the full route for submitted nested-validation tables")
        for name in [
            "item_selection_cv_folds.csv",
            "item_selection_cv_summary.csv",
            "item_selection_stability.csv",
            "tableS_nested_selective_escalation_cv_folds.csv",
            "tableS_nested_selective_escalation_cv_summary.csv",
        ]:
            archived = static_dir.parent.parent / "outputs" / "tables" / name
            if archived.exists():
                pd.read_csv(archived).to_csv(tables/name, index=False)
    else:
        log_step("Running fully nested item-selection CV and nested workflow-threshold CV")
        nested_folds, nested_summary, nested_stability = fully_nested_region_stratified_cv(d)
        nested_folds.to_csv(tables/"item_selection_cv_folds.csv", index=False)
        nested_summary.to_csv(tables/"item_selection_cv_summary.csv", index=False)
        nested_stability.to_csv(tables/"item_selection_stability.csv", index=False)
        nested_workflow_folds, nested_workflow_summary = nested_selective_escalation_cv(d)
        nested_workflow_folds.to_csv(tables/"tableS_nested_selective_escalation_cv_folds.csv", index=False)
        nested_workflow_summary.to_csv(tables/"tableS_nested_selective_escalation_cv_summary.csv", index=False)

    response_pattern_audit_table(d).to_csv(tables/"table14_response_pattern_audit.csv", index=False)
    tripod_ai_reporting_checklist().to_csv(tables/"tableS_tripod_ai_informed_reporting_checklist.csv", index=False)
    deployment_card().to_csv(tables/"tableS_intended_use_deployment_card.csv", index=False)

    log_step("Writing pooled CV workbook inputs and reproducibility metadata")
    folds, cv = pooled_cv(d, selected)
    folds.to_csv(tables/"pooled_region_stratified_cv_folds.csv", index=False)
    cv.to_csv(tables/"pooled_region_stratified_cv_summary.csv", index=False)
    with open(args.out/"analysis_run_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "data_file_name": args.data.name,
            "data_source": "Mendeley Data DOI: 10.17632/btzgmr2rt2.1; place Data_collegestudent.csv locally and pass its path with --data",
            "random_repeats": args.random_repeats,
            "random_baseline_source": random_baseline_source,
            "bootstrap_ci": args.bootstrap_ci,
            "quick_mode": bool(args.quick),
            "skip_nested_cv": bool(args.skip_nested_cv),
            "random_state": RANDOM_STATE,
            "operational_lasso_order": "code/static/lasso_rank_hunan.csv",
            "archived_simulation_note": "By default, run_all.py and analysis.py reproduce archived random item-set and selection-stability simulation tables from code/static. Use --recompute-random-baseline to recompute random item-set distributions; selection-stability simulations are archived static outputs in this release.",
            "package_version": "1.0.0",
            "run_mode": "quick" if args.quick else ("custom" if args.skip_nested_cv or args.bootstrap_ci != 500 else "full"),
            "tree_based_sensitivity_default": "not run by run_all.py unless --include-tree is supplied"
        }, f, indent=2)
    write_environment(args.out, args.quick, args.skip_nested_cv, args.bootstrap_ci)
    print(f"Done. Results written to {args.out}")

if __name__ == "__main__":
    main()
