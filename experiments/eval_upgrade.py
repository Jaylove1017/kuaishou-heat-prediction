# -*- coding: utf-8 -*-
"""评估口径升级实验（论文第 5 章用）

解决三件事：
1. per-day 评估：把「全局排序取 top-k」改为「按天分别圈选再汇总」——业务上正确的口径
2. 精确率实算：原 rec() 只输出召回，本脚本同时输出 precision
3. 出图：模型 vs 基线的 top-k 对比曲线、跃升口径 PR 曲线

输出：
  outputs/exp_eval_upgrade.json
  charts/exp_topk_compare.png
  charts/exp_pr_breakout.png

与既有脚本的关系：
  - 特征构造 / 样本切分 / 标签定义 与 experiments/baseline_vs_model.py 完全一致
  - 唯一改动是评估函数（新增 per-day + precision），因此结果可与旧口径直接对照
"""
import sys
import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_TAIL_DAYS = 14
TOP_Q = 0.75
OUT = os.path.join(PROJ, "outputs")
CH = os.path.join(PROJ, "charts")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CH, exist_ok=True)

# ---------------------------------------------------------------- 数据与特征
print("[1] 读取 CSV 并构造特征 ...")
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

# 标签 A：次日点赞进入当日前 25%
day_q = df.groupby("next_d")["next_like"].transform(lambda s: s.quantile(TOP_Q))
df["label_abs"] = (df["next_like"] >= day_q).astype(int)
# 标签 B：相对自身近 7 日均值的跃升
base7 = df["roll7_like"].replace(0, np.nan)
df["label_break"] = ((df["next_like"] >= 2 * base7) & (df["next_like"] > 0)).astype(int)
df["label_break"] = df["label_break"].where(base7.notna(), 0).astype(int)

FEATURES = ["dur_log", "is_ad", "up_type_idx", "tag_idx", "dow",
            "play_user_num", "like_user_num", "complete_play_cnt", "like_rate", "complete_rate",
            "lag1_like", "lag2_like", "lag3_like", "lag1_playuser", "lag2_playuser", "lag3_playuser",
            "lag1_likerate", "roll7_like", "roll7_playuser", "roll7_complete", "n_prior_days", "gap_days"]
ds = df.dropna(subset=FEATURES).copy()
ds = ds.sort_values(["d", "video_id"]).reset_index(drop=True)

day_max = ds["d"].max()
split_d = int((pd.to_datetime(str(day_max), format="%Y%m%d")
               - pd.Timedelta(days=SPLIT_TAIL_DAYS)).strftime("%Y%m%d"))
tr, te = ds[ds["d"] <= split_d], ds[ds["d"] > split_d]
print(f"    样本 {len(ds):,}  train {len(tr):,}  test {len(te):,}  split={split_d}")
print(f"    测试集天数 {te['d'].nunique()}，每日样本 "
      f"min={te.groupby('d').size().min()} / max={te.groupby('d').size().max()}")


# ---------------------------------------------------------------- 评估函数
def eval_global(scores, y, k):
    """全局排序：把全部测试样本混在一起取 top-k%（原口径）"""
    n = max(int(len(y) * k), 1)
    idx = np.argsort(-np.asarray(scores, dtype=float))[:n]
    hits = int(y[idx].sum())
    return dict(recall=hits / max(int(y.sum()), 1) * 100,
                precision=hits / n * 100, n_sel=n, hits=hits)


def eval_per_day(scores, y, days, k):
    """按天分别圈选：每天取当天样本的 top-k%，再汇总（业务正确口径）"""
    hits = 0
    n_sel = 0
    for d in np.unique(days):
        m = days == d
        s = np.asarray(scores, dtype=float)[m]
        yy = np.asarray(y)[m]
        n = max(int(m.sum() * k), 1)
        idx = np.argsort(-s)[:n]
        hits += int(yy[idx].sum())
        n_sel += n
    return dict(recall=hits / max(int(np.sum(y)), 1) * 100,
                precision=hits / max(n_sel, 1) * 100, n_sel=n_sel, hits=hits)


# ---------------------------------------------------------------- 训练
print("[2] 训练模型 ...")
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import xgboost as xgb

Xtr, Xte = tr[FEATURES], te[FEATURES]
days_te = te["d"].values
SCENARIOS = [
    dict(key="abs", label="label_abs", name="绝对热度（次日进前25%）", naive_col="like_user_num"),
    dict(key="break", label="label_break", name="相对跃升（≥自身7日均值2倍）", naive_col="roll7_like"),
]
KS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
result = {"split_day": int(split_d), "test_rows": int(len(te)),
          "test_days": int(te["d"].nunique()), "features": len(FEATURES), "scenarios": {}}

for sc in SCENARIOS:
    ytr = tr[sc["label"]].values
    yte = te[sc["label"]].values
    print(f"\n[3] 口径：{sc['name']}  正样本率 test={yte.mean()*100:.2f}%")

    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced"))
    lr.fit(Xtr, ytr)
    p_lr = lr.predict_proba(Xte)[:, 1]

    xm = xgb.XGBClassifier(n_estimators=500, learning_rate=0.06, max_depth=6,
                           subsample=0.9, colsample_bytree=0.9, eval_metric="auc",
                           tree_method="hist", n_jobs=4, random_state=42)
    xm.fit(Xtr, ytr)
    p_xgb = xm.predict_proba(Xte)[:, 1]

    naive = te[sc["naive_col"]].fillna(-1).values.astype(float)

    r = dict(name=sc["name"], pos_rate=round(float(yte.mean()) * 100, 2),
             auc=dict(lr=round(float(roc_auc_score(yte, p_lr)), 4),
                      xgb=round(float(roc_auc_score(yte, p_xgb)), 4),
                      naive=round(float(roc_auc_score(yte, naive)), 4)),
             ap=dict(lr=round(float(average_precision_score(yte, p_lr)), 4),
                     xgb=round(float(average_precision_score(yte, p_xgb)), 4),
                     naive=round(float(average_precision_score(yte, naive)), 4)),
             global_topk={}, perday_topk={})
    print(f"    AUC  LR={r['auc']['lr']:.4f}  XGB={r['auc']['xgb']:.4f}  基线={r['auc']['naive']:.4f}")

    for k in KS:
        km = f"{int(k*100)}"
        gx = eval_global(p_xgb, yte, k)
        gn = eval_global(naive, yte, k)
        px = eval_per_day(p_xgb, yte, days_te, k)
        pn = eval_per_day(naive, yte, days_te, k)
        r["global_topk"][km] = dict(model=gx, naive=gn,
                                    lift_recall=round(gx["recall"] - gn["recall"], 2),
                                    lift_precision=round(gx["precision"] - gn["precision"], 2))
        r["perday_topk"][km] = dict(model=px, naive=pn,
                                    lift_recall=round(px["recall"] - pn["recall"], 2),
                                    lift_precision=round(px["precision"] - pn["precision"], 2))
    for k in (0.10, 0.20, 0.30):
        km = f"{int(k*100)}"
        a, b = r["global_topk"][km], r["perday_topk"][km]
        print(f"    top{km}%  全局: 模型 R={a['model']['recall']:.2f}% P={a['model']['precision']:.2f}% "
              f"| 基线 R={a['naive']['recall']:.2f}% P={a['naive']['precision']:.2f}%")
        print(f"          按天: 模型 R={b['model']['recall']:.2f}% P={b['model']['precision']:.2f}% "
              f"| 基线 R={b['naive']['recall']:.2f}% P={b['naive']['precision']:.2f}%")

    r["_curves"] = dict(p_lr=p_lr, p_xgb=p_xgb, naive=naive, y=yte)
    result["scenarios"][sc["key"]] = r

# ---------------------------------------------------------------- 落盘（去掉数组）
serializable = json.loads(json.dumps(result, default=str))
for k in serializable["scenarios"]:
    serializable["scenarios"][k].pop("_curves", None)
with open(os.path.join(OUT, "exp_eval_upgrade.json"), "w", encoding="utf-8") as f:
    json.dump(serializable, f, ensure_ascii=False, indent=2)
print("\n[4] 已写 outputs/exp_eval_upgrade.json")

# ---------------------------------------------------------------- 图 1：top-k 对比
print("[5] 画图 ...")
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9))
ks = [int(k * 100) for k in KS]

# 左：绝对口径下「模型 − 惯性基线」的召回增益（绝对值曲线会重合，改画差值才看得见）
ax = axes[0]
r_abs = result["scenarios"]["abs"]
lift_global = [r_abs["global_topk"][str(k)]["lift_recall"] for k in ks]
lift_perday = [r_abs["perday_topk"][str(k)]["lift_recall"] for k in ks]
ax.plot(ks, lift_global, "o-", color="#C0392B", lw=2, label="模型 − 基线（全局排序）")
ax.plot(ks, lift_perday, "s--", color="#1F4E79", lw=2, label="模型 − 基线（按天圈选）")
ax.axhline(0, color="black", lw=1.3)
ax.fill_between(ks, 0, lift_global, where=[v < 0 for v in lift_global],
                color="#C0392B", alpha=.12)
ax.set(title="绝对热度口径：模型相对惯性基线的召回增益",
       xlabel="圈选比例 Top-k（%）", ylabel="召回增益（百分点 pp）")
ax.grid(alpha=.3)
ax.legend(fontsize=9, loc="lower left")
ax.text(0.02, 0.93, "0 线以下 = 模型不如「按当日点赞排序」",
        transform=ax.transAxes, ha="left", fontsize=8.5, color="#7a7a7a")

# 右：跃升口径的召回曲线（差距太大，用绝对曲线更有冲击力）
ax = axes[1]
r_brk = result["scenarios"]["break"]
px = [r_brk["perday_topk"][str(k)]["model"]["recall"] for k in ks]
pn = [r_brk["perday_topk"][str(k)]["naive"]["recall"] for k in ks]
ax.plot(ks, px, "^-", color="#1F4E79", lw=2.2, label="模型（按天圈选）")
ax.plot(ks, pn, "v--", color="#E67E22", lw=1.9, label="惯性基线：近7日均值排序")
ax.plot([5, 30], [5, 30], "k:", lw=1, alpha=.45, label="随机基线")
ax.set(title="相对跃升口径：top-k 圈选召回率",
       xlabel="圈选比例 Top-k（%）", ylabel="命中率 / 召回（%）")
ax.grid(alpha=.3)
ax.legend(fontsize=9, loc="center right")
fig.tight_layout()
fig.savefig(os.path.join(CH, "exp_topk_compare.png"), dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- 图 2：跃升口径 PR
r = result["scenarios"]["break"]
y = r["_curves"]["y"]
fig, ax = plt.subplots(figsize=(7.2, 5.0))
for p, c, lab in ((r["_curves"]["p_xgb"], "#1F4E79", "模型 XGBoost"),
                  (r["_curves"]["p_lr"], "#16A085", "模型 逻辑回归"),
                  (r["_curves"]["naive"], "#E67E22", "惯性基线（近7日均值）")):
    pr, rc, _ = precision_recall_curve(y, p)
    ax.plot(rc, pr, lw=1.9, color=c, label=f"{lab} (AP={average_precision_score(y, p):.3f})")
ax.axhline(y.mean(), color="gray", ls=":", lw=1.2, label=f"随机基线 (正样本率 {y.mean()*100:.1f}%)")
pt = r["perday_topk"]["20"]["model"]
ax.scatter([pt["recall"] / 100], [pt["precision"] / 100], s=110, color="#C0392B", zorder=6)
ax.annotate(f"按天圈选 top20%\n召回 {pt['recall']:.1f}% / 精确 {pt['precision']:.1f}%",
            (pt["recall"] / 100, pt["precision"] / 100),
            textcoords="offset points", xytext=(-96, 26), color="#C0392B", fontsize=8.5)
ax.set(title="相对跃升口径：Precision-Recall 曲线（时间切分测试集）",
       xlabel="召回 Recall", ylabel="精确 Precision", xlim=(0, 1), ylim=(0, 1))
ax.grid(alpha=.3)
ax.legend(fontsize=8.5, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(CH, "exp_pr_breakout.png"), dpi=160)
plt.close(fig)

print("    已写 charts/exp_topk_compare.png, charts/exp_pr_breakout.png")
print("\n完成。")
