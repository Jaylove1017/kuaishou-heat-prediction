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
  - 特征构造 / 样本切分 / 标签定义 统一 import 自 src/features.py，
    不再各存一份副本（五个副本已漂移成两条数据通路）
  - 唯一改动是评估函数（新增 per-day + precision），因此结果可与旧口径直接对照
  - 本脚本的 exp_eval_upgrade.json 是 README 全部 top-k 数字的唯一来源
"""
import sys
import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import features as F

sys.stdout.reconfigure(encoding="utf-8")
F.setup_style()

PROJ = F.PROJ
SPLIT_TAIL_DAYS = F.SPLIT_TAIL_DAYS
TOP_Q = F.TOP_Q
OUT = os.path.join(PROJ, "outputs")
CH = os.path.join(PROJ, "charts")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CH, exist_ok=True)

# ---------------------------------------------------------------- 数据与特征
print("[1] 读取 CSV 并构造特征 ...")
df = F.filter_active(F.load_daily_csv())
ds = F.build_dataset(df)
ds = ds.sort_values(["d", "video_id"]).reset_index(drop=True)
FEATURES = F.FEATURES

tr, te, split_d = F.split(ds)
print(f"    样本 {len(ds):,}  train {len(tr):,}  test {len(te):,}  split={split_d}")
print(f"    测试集天数 {te['d'].nunique()}，每日样本 "
      f"min={te.groupby('d').size().min()} / max={te.groupby('d').size().max()}")


# ---------------------------------------------------------------- 评估函数
# 实现在 src/features.py（与 ablation.py / significance.py 共用同一份），
# 避免「每个脚本各写一遍 top-k」然后再也说不清哪两个数字可比。
eval_global = F.eval_global
eval_per_day = F.eval_per_day


# ---------------------------------------------------------------- 训练
print("[2] 训练模型 ...")
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

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

    lr = F.make_lr()
    lr.fit(Xtr, ytr)
    p_lr = lr.predict_proba(Xte)[:, 1]

    xm = F.make_xgb()
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
