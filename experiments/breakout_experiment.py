# -*- coding: utf-8 -*-
"""延伸实验：把标签从"绝对高潜"换成"相对自身的跃升(breakout)"，
检验模型在惯性解释不了的地方是否有真增量——用于回答"为什么不直接按当日点赞排序"。
"""
import sys, os, json
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录（可移植）

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

FEATURES = ["dur_log", "is_ad", "up_type_idx", "tag_idx", "dow",
            "play_user_num", "like_user_num", "complete_play_cnt", "like_rate", "complete_rate",
            "lag1_like", "lag2_like", "lag3_like", "lag1_playuser", "lag2_playuser", "lag3_playuser",
            "lag1_likerate", "roll7_like", "roll7_playuser", "roll7_complete", "n_prior_days", "gap_days"]
ds = df.dropna(subset=FEATURES).copy()

# 标签 A（原口径）：次日点赞进入当日活跃视频前 25%
ds["label_abs"] = (ds["next_like"] >= ds.groupby("next_d")["next_like"].transform(lambda s: s.quantile(.75))).astype(int)
# 标签 B（跃升口径）：次日点赞 ≥ 自身近7日（不含当日）均值的 2 倍，且次日有点赞
base7 = ds["roll7_like"].replace(0, np.nan)
ds["label_break"] = ((ds["next_like"] >= 2 * base7) & (ds["next_like"] > 0)).astype(int)
ds["label_break"] = ds["label_break"].where(base7.notna(), 0).astype(int)

day_max = ds["d"].max()
split_d = int((pd.to_datetime(str(day_max), format="%Y%m%d") - pd.Timedelta(days=14)).strftime("%Y%m%d"))
tr, te = ds[ds["d"] <= split_d], ds[ds["d"] > split_d]

import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score


def rec(s, y, k):
    order = np.argsort(-np.asarray(pd.Series(s).fillna(-1).values, float))
    top = order[: int(len(y) * k)]
    return y[top].sum() / max(y.sum(), 1) * 100


def run(labelcol, naive_col, naive_name, feats=FEATURES):
    Xtr, ytr = tr[feats], tr[labelcol].values
    Xte, yte = te[feats], te[labelcol].values
    if yte.sum() == 0:
        print("无正样本"); return None
    m = xgb.XGBClassifier(n_estimators=500, learning_rate=0.06, max_depth=6, subsample=0.9,
                          colsample_bytree=0.9, eval_metric="auc", tree_method="hist",
                          n_jobs=-1, random_state=42)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    naive = te[naive_col].fillna(-1).values
    print(f"\n=== 标签：{labelcol}  (正样本率 test {yte.mean()*100:.2f}%) ===")
    print(f"  模型 XGB      AUC={roc_auc_score(yte,p):.4f}  AP={average_precision_score(yte,p):.4f}  "
          f"top20%召回={rec(p,yte,.2):.2f}%（随机基线 {yte.mean()*100:.2f}%）")
    print(f"  朴素 {naive_name:<12} AUC={roc_auc_score(yte,naive):.4f} "
          f" top20%召回={rec(naive,yte,.2):.2f}%")
    out = dict(pos_rate=round(float(yte.mean()) * 100, 2),
               auc_model=round(float(roc_auc_score(yte, p)), 4),
               auc_naive=round(float(roc_auc_score(yte, naive)), 4),
               top20_model=round(rec(p, yte, .2), 2), top20_naive=round(rec(naive, yte, .2), 2))
    for k in (.05, .10, .20, .30):
        out[f"top{int(k*100)}_model"] = round(rec(p, yte, k), 2)
        out[f"top{int(k*100)}_naive"] = round(rec(naive, yte, k), 2)
    print("  top-k 对照:", {f"top{int(k*100)}%": f"模型{rec(p,yte,k):.1f}/朴素{rec(naive,yte,k):.1f}" for k in (.05, .10, .20, .30)})
    return out


res = {}
print(f"样本 {len(ds):,}  train {len(tr):,}  test {len(te):,}  split={split_d}")
res["绝对高潜(原口径)"] = run("label_abs", "like_user_num", "当日点赞")
res["相对跃升(breakout)"] = run("label_break", "roll7_like", "近7日均值")
res["相对跃升(breakout)"] = res.get("相对跃升(breakout)") or {}

# 跃升口径下，只用历史特征（去掉当日 like_user_num/play_user_num 等当日热度），
# 看纯"内容属性+趋势"能预测多少 —— 这才是模型不可替代的部分
feats_nocur = [f for f in FEATURES if f not in
               ("like_user_num", "play_user_num", "complete_play_cnt", "like_rate", "complete_rate")]
print("\n[跃升口径] 去掉所有'当日热度'特征，只用内容属性+滞后/滚动趋势：")
print("  参与特征:", feats_nocur)
r2 = run("label_break", "roll7_like", "近7日均值(全特征基线)", feats=feats_nocur)
res["跃升+去当日热度"] = r2

with open(os.path.join(PROJ, "outputs", "breakout_experiment.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print("\n已写 outputs/breakout_experiment.json")
