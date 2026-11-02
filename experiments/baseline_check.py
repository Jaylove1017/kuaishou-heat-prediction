# -*- coding: utf-8 -*-
"""朴素基线对照：如果不用模型，直接按"当日点赞量"排序圈选高潜，能命中多少？

⚠️ 本脚本为【早期探查版，口径不严格，结论不可直接引用】⚠️
   缺陷 1：模型列（77.02% / AUC 0.9820 及 r_model 字典）是从 metrics.json **硬编码**写入的，
           并非在本脚本的测试集上重算；
   缺陷 2：本脚本未做 dropna(subset=FEATURES)，测试集为 98,672 行，
           而 model_analysis.py 的测试集为 92,735 行 —— 两者不是同一批样本。
   后果：本脚本输出的"提升 +2.33pp"是跨样本比较的产物，**不成立**。
   正确结论请以 experiments/baseline_vs_model.py（严格同口径）为准。
"""
import sys, os
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录（可移植）
SPLIT_TAIL_DAYS = 14
TOP_Q = 0.75

df = pd.read_csv(os.path.join(PROJ, "data", "clean", "video_daily.csv"), low_memory=False)
df = df.rename(columns={"date": "d"})
df = df[(df["play_cnt"] > 0) & (df["d"] >= 20200705)].copy()
print("活跃视频日:", len(df))

g = df.sort_values("d").groupby("video_id")
for lag in (1, 2, 3):
    df[f"lag{lag}_like"] = g["like_user_num"].shift(lag)
df["roll7_like"] = g["like_user_num"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
df["next_d"] = g["d"].shift(-1)
df["next_like"] = g["like_user_num"].shift(-1)
df = df[df["next_d"].notna()].copy()
df["next_d"] = df["next_d"].astype(int)
day_q = df.groupby("next_d")["next_like"].transform(lambda s: s.quantile(TOP_Q))
df["label"] = (df["next_like"] >= day_q).astype(int)

day_max = df["d"].max()
split_d = int((pd.to_datetime(str(day_max), format="%Y%m%d")
               - pd.Timedelta(days=SPLIT_TAIL_DAYS)).strftime("%Y%m%d"))
te = df[df["d"] > split_d].copy()
print(f"split_day={split_d}  test={len(te):,} 行  正样本率={te['label'].mean()*100:.2f}%")


def topk_recall(scores, y, k=0.20):
    order = np.argsort(-np.asarray(scores, dtype=float))
    n = len(y)
    top = order[: int(n * k)]
    return y[top].sum() / max(y.sum(), 1) * 100, y[top].mean() * 100


y = te["label"].values
cands = {
    "模型 XGBoost(已知)": None,
    "朴素：当日 like_user_num": te["like_user_num"].values,
    "朴素：昨日 lag1_like": te["lag1_like"].values,
    "朴素：近7日均值 roll7_like": te["roll7_like"].values,
    "朴素：当日 play_user_num": te["play_user_num"].values,
}
from sklearn.metrics import roc_auc_score
print(f"\n{'排序口径':<26}{'top20%召回':>10}{'top20%精度':>11}{'AUC':>9}")
print("-" * 58)
for name, s in cands.items():
    if s is None:
        print(f"{name:<26}{77.02:>9.2f}%{'':>10}{0.9820:>9.4f}")
        continue
    s = pd.Series(s).fillna(-1).values
    r, p = topk_recall(s, y)
    auc = roc_auc_score(y, s)
    print(f"{name:<26}{r:>9.2f}%{p:>10.2f}%{auc:>9.4f}")

# 所有朴素基线里最强的一个 vs 模型：详细 topk
best = te["like_user_num"].fillna(-1).values
print("\n细致对照（top-k 召回）：")
for k in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
    r_model = {0.05: 20.03, 0.10: 39.95, 0.15: 59.53, 0.20: 77.05, 0.25: 88.66, 0.30: 94.21}
    rn, _ = topk_recall(best, y, k)
    print(f"  top{int(k*100):>2}%   模型 {r_model[k]:>6.2f}%   朴素(当日点赞) {rn:>6.2f}%   提升 {r_model[k]-rn:>+6.2f}pp")
