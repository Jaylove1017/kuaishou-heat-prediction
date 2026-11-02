# -*- coding: utf-8 -*-
"""严格同口径对照：严格复刻 model_analysis.py 的样本构造/切分/参数，
在同一测试集上比"模型 vs 朴素按当日点赞排序"，并做历史长度分段的增益拆解。
"""
import sys, os, json
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录（可移植）
SPLIT_TAIL_DAYS, TOP_Q = 14, 0.75

df = pd.read_csv(os.path.join(PROJ, "data", "clean", "video_daily.csv"), low_memory=False)
df = df.rename(columns={"date": "d"})
df = df[(df["play_cnt"] > 0) & (df["d"] >= 20200705)].copy()
df["date"] = pd.to_datetime(df["d"].astype(str), format="%Y%m%d")
df["dur_log"] = np.log1p(df["video_duration_ms"].fillna(0) / 1000.0)
df["is_ad"] = (df["video_type"] == "AD").astype(int)
df["up_type_idx"] = df["upload_type"].fillna("UNKNOWN").astype("category").cat.codes
df["tag_idx"] = df["video_tag_name"].fillna("_NA_").astype("category").cat.codes
df["dow"] = df["date"].dt.weekday
df["like_rate"] = (df["like_user_num"] / df["play_user_num"].replace(0, np.nan)).fillna(0)
df["complete_rate"] = (df["complete_play_cnt"] / df["play_cnt"].replace(0, np.nan)).fillna(0)

g = df.sort_values("d").groupby("video_id")
for lag in (1, 2, 3):
    df[f"lag{lag}_like"] = g["like_user_num"].shift(lag)
    df[f"lag{lag}_playuser"] = g["play_user_num"].shift(lag)
df["lag1_likerate"] = g["like_rate"].shift(1)
df["roll7_like"] = g["like_user_num"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
df["roll7_playuser"] = g["play_user_num"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
df["roll7_complete"] = g["complete_play_cnt"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
df["n_prior_days"] = g.cumcount()
df["gap_days"] = g["d"].diff().fillna(99)
df["next_d"] = g["d"].shift(-1)
df["next_like"] = g["like_user_num"].shift(-1)
df = df[df["next_d"].notna()].copy()
df["next_d"] = df["next_d"].astype(int)
day_q = df.groupby("next_d")["next_like"].transform(lambda s: s.quantile(TOP_Q))
df["label"] = (df["next_like"] >= day_q).astype(int)

FEATURES = ["dur_log", "is_ad", "up_type_idx", "tag_idx", "dow",
            "play_user_num", "like_user_num", "complete_play_cnt", "like_rate", "complete_rate",
            "lag1_like", "lag2_like", "lag3_like", "lag1_playuser", "lag2_playuser", "lag3_playuser",
            "lag1_likerate", "roll7_like", "roll7_playuser", "roll7_complete", "n_prior_days", "gap_days"]
ds = df.dropna(subset=FEATURES)
day_max = ds["d"].max()
split_d = int((pd.to_datetime(str(day_max), format="%Y%m%d") - pd.Timedelta(days=SPLIT_TAIL_DAYS)).strftime("%Y%m%d"))
tr, te = ds[ds["d"] <= split_d], ds[ds["d"] > split_d]
print(f"样本 {len(ds):,}  train {len(tr):,}  test {len(te):,}  split={split_d}")

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

Xtr, ytr = tr[FEATURES], tr["label"].values
Xte, yte = te[FEATURES], te["label"].values
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced"))
lr.fit(Xtr, ytr)
p_lr = lr.predict_proba(Xte)[:, 1]
xm = xgb.XGBClassifier(n_estimators=500, learning_rate=0.06, max_depth=6, subsample=0.9,
                       colsample_bytree=0.9, eval_metric="auc", tree_method="hist",
                       n_jobs=-1, random_state=42)
xm.fit(Xtr, ytr)
p_xgb = xm.predict_proba(Xte)[:, 1]
print(f"[复现校验] LR AUC={roc_auc_score(yte,p_lr):.4f}  XGB AUC={roc_auc_score(yte,p_xgb):.4f} (目标 0.9458 / 0.9825)")

naive = te["like_user_num"].fillna(-1).values


def rec(scores, y, k):
    order = np.argsort(-np.asarray(scores, float))
    top = order[: int(len(y) * k)]
    return y[top].sum() / max(y.sum(), 1) * 100


print(f"\n{'top-k':<8}{'模型XGB':>10}{'朴素当日点赞':>14}{'增益':>9}")
print("-" * 42)
res = {}
for k in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
    a, b = rec(p_xgb, yte, k), rec(naive, yte, k)
    res[k] = dict(model=round(a, 2), naive=round(b, 2), lift=round(a - b, 2))
    print(f"top{int(k*100):>2}%{'':<2}{a:>9.2f}%{b:>13.2f}%{a-b:>+8.2f}pp")
print(f"\nAUC: 模型 {roc_auc_score(yte,p_xgb):.4f} | 朴素当日点赞 {roc_auc_score(yte,naive):.4f}"
      f" | 朴素昨日 {roc_auc_score(yte, te['lag1_like'].fillna(-1).values):.4f}")

print("\n分段增益（top20% 召回，按历史活跃天数分组）:")
print(f"{'历史长度':<14}{'样本数':>9}{'正样本率':>9}{'模型':>9}{'朴素':>9}{'增益':>9}")
print("-" * 60)
seg_res = {}
bins = [(-1, 0, "首次活跃(0天)"), (0, 2, "1-2天"), (2, 6, "3-6天"), (6, 10**9, "7天以上")]
for lo, hi, name in bins:
    m = (te["n_prior_days"] > lo) & (te["n_prior_days"] <= hi)
    if m.sum() < 200:
        continue
    y = yte[m.values]
    a, b = rec(p_xgb[m.values], y, .2), rec(naive[m.values], y, .2)
    seg_res[name] = dict(n=int(m.sum()), pos=round(float(y.mean()) * 100, 2),
                         model=round(a, 2), naive=round(b, 2), lift=round(a - b, 2))
    print(f"{name:<14}{m.sum():>9,}{y.mean()*100:>8.2f}%{a:>8.2f}%{b:>8.2f}%{a-b:>+8.2f}pp")

imp = sorted(zip(FEATURES, xm.feature_importances_), key=lambda t: -t[1])
print("\n前 6 特征:", [(f, round(float(v), 4)) for f, v in imp[:6]])

with open(os.path.join(PROJ, "outputs", "baseline_vs_model.json"), "w", encoding="utf-8") as f:
    json.dump(dict(test_rows=int(len(te)), split_day=int(split_d),
                   auc_model=round(float(roc_auc_score(yte, p_xgb)), 4),
                   auc_naive_today=round(float(roc_auc_score(yte, naive)), 4),
                   auc_naive_lag1=round(float(roc_auc_score(yte, te["lag1_like"].fillna(-1).values)), 4),
                   topk=res, by_history=seg_res), f, ensure_ascii=False, indent=2)
print("\n已写 outputs/baseline_vs_model.json")
