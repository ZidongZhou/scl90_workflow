#!/usr/bin/env python3
"""Tree-based item-ranking sensitivity analyses for the SCL-90 PLOS ONE study."""
from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

ITEMS = [f"S{i}" for i in range(1, 91)]
RANDOM_STATE = 20260610
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")


def load_and_qc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["location", "gender", "total", *ITEMS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    locations = set(df["location"].astype(int).unique().tolist())
    if locations != {1, 2}:
        raise ValueError(f"Expected location codes exactly {{1, 2}}; observed {sorted(locations)}")
    invalid = (df[ITEMS].lt(0).any(axis=1) | df[ITEMS].gt(4).any(axis=1) | ~df["gender"].isin([1,2]) | ((df[ITEMS].sum(axis=1)+90) != df["total"]))
    d = df.loc[~invalid].copy()
    d["label_ge160"] = (d["total"] >= 160).astype(int)
    d["region"] = d["location"].map({1: "Shandong", 2: "Hunan"})
    return d


def youden(y, p):
    best, bestj = 0.5, -1
    for t in np.quantile(p, np.linspace(0,1,251)):
        pred = (p>=t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
        j = tp/(tp+fn) + tn/(tn+fp) - 1
        if j > bestj:
            bestj, best = j, float(t)
    return best


def evaluate(y, p, thr):
    pred=(p>=thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    order=np.argsort(-np.asarray(p), kind="stable"); idx=order[:int(round(.10*len(y)))]
    return {"roc_auc": roc_auc_score(y,p), "pr_auc": average_precision_score(y,p), "brier": brier_score_loss(y,p), "sensitivity": tp/(tp+fn), "specificity": tn/(tn+fp), "top10_recall": y[idx].sum()/y.sum(), "top10_precision": y[idx].mean()}


def fit_eval(h, s, selected_items):
    sc=StandardScaler().fit(h[selected_items])
    Xh=sc.transform(h[selected_items]); Xs=sc.transform(s[selected_items])
    y=h.label_ge160.to_numpy(); ys=s.label_ge160.to_numpy()
    clf=LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE).fit(Xh,y)
    ph=clf.predict_proba(Xh)[:,1]; ps=clf.predict_proba(Xs)[:,1]
    return evaluate(ys, ps, youden(y, ph))


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", default=Path("outputs_run"), type=Path)
    args=ap.parse_args()
    tables=args.out/"tables"; tables.mkdir(parents=True, exist_ok=True)
    d=load_and_qc(args.data); h=d[d.region=="Hunan"].copy(); s=d[d.region=="Shandong"].copy()
    rows=[]
    try:
        from lightgbm import LGBMClassifier
    except ImportError as e:
        rows.append({"method": "LightGBM unavailable", "selected_items": "", "note": f"Optional dependency not installed: {e}"})
    else:
        lgb = LGBMClassifier(n_estimators=50, learning_rate=0.06, num_leaves=15, max_depth=5,
                             class_weight="balanced", random_state=RANDOM_STATE, verbose=-1, n_jobs=1)
        lgb.fit(h[ITEMS], h.label_ge160)
        rank = pd.DataFrame({"item": ITEMS, "importance_gain": lgb.booster_.feature_importance(importance_type="gain")}).sort_values("importance_gain", ascending=False)
        rank.to_csv(tables/"lightgbm_gain_item_ranking_hunan_all90.csv", index=False)
        items = rank.item.head(15).tolist()
        rows.append({"method": "LightGBM gain ranking + logistic 15-item set", "selected_items": ",".join(items), **fit_eval(h, s, items)})

    try:
        from xgboost import XGBClassifier, DMatrix
    except ImportError as e:
        rows.append({"method": "XGBoost unavailable", "selected_items": "", "note": f"Optional dependency not installed: {e}"})
    else:
        xgb = XGBClassifier(n_estimators=40, max_depth=3, learning_rate=0.08, subsample=.8,
                            colsample_bytree=.8, objective="binary:logistic", eval_metric="logloss",
                            random_state=RANDOM_STATE, n_jobs=1, tree_method="hist",
                            scale_pos_weight=(1-h.label_ge160.mean())/h.label_ge160.mean())
        xgb.fit(h[ITEMS], h.label_ge160)
        sample = h[ITEMS].sample(n=min(600, len(h)), random_state=RANDOM_STATE)
        contrib = xgb.get_booster().predict(DMatrix(sample), pred_contribs=True)
        imp = np.abs(contrib[:, :-1]).mean(axis=0)
        rank = pd.DataFrame({"item": ITEMS, "mean_abs_xgboost_contribution": imp}).sort_values("mean_abs_xgboost_contribution", ascending=False)
        rank.to_csv(tables/"xgboost_contribution_item_ranking_hunan_all90.csv", index=False)
        items = rank.item.head(15).tolist()
        rows.append({"method": "XGBoost contribution-score ranking + logistic 15-item set", "selected_items": ",".join(items), **fit_eval(h, s, items)})
    pd.DataFrame(rows).to_csv(tables/"tableS_tree_based_selection_sensitivity.csv", index=False)
    print("Wrote tree-based sensitivity outputs to", tables)

if __name__ == "__main__":
    main()
