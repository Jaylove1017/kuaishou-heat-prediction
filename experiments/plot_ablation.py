# -*- coding: utf-8 -*-
"""消融实验出图（只读 outputs/exp_ablation.json，不重跑模型）

与 ablation.py 分离的原因：论文阶段需要反复调图，重跑模型每次近 2 分钟，
而画图只需 1 秒。数据一旦定稿，图可以单独迭代。

用法：
    python experiments/plot_ablation.py

输出：
  charts/exp_ablation_auc.png      8 组特征 × 2 口径的 AUC 对比
  charts/exp_ablation_roll7.png    剔除 roll7_like 后的 AUC 下降量（避免截断轴误导）
"""
import sys
import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CH = os.path.join(PROJ, "charts")
os.makedirs(CH, exist_ok=True)

with open(os.path.join(PROJ, "outputs", "exp_ablation.json"), "r", encoding="utf-8") as f:
    R = json.load(f)

gids = [g["id"] for g in R["groups"]]
SHORT = {
    "G1": "G1 仅内容属性 (5)",
    "G2": "G2 仅当日热度 (5)",
    "G3": "G3 仅惯性趋势 (12)",
    "G4": "G4 内容＋当日热度 (10)",
    "G5": "G5 内容＋惯性 (17)",
    "G6": "G6 内容＋惯性−roll7 (16)",
    "G7": "G7 全部 22 维",
    "G8": "G8 全部−roll7 (21)",
}
PANELS = [("abs", "绝对热度口径（次日进前 25%）"),
          ("break", "相对跃升口径（≥自身 7 日均值 2 倍）")]

# ================================================================ 图 1
fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8))
for ax, (key, title) in zip(axes, PANELS):
    b = R["scenarios"][key]
    vals = [b["groups"][g]["auc_xgb"] for g in gids]
    lrs = [b["groups"][g]["auc_lr"] for g in gids]
    ypos = np.arange(len(gids))[::-1]
    colors = ["#C0392B" if g in ("G7", "G8") else "#95A5A6" for g in gids]
    ax.barh(ypos, vals, color=colors, height=.6, alpha=.9)

    base = b["baseline"]["auc"]
    xmin = min(min(vals), min(lrs), base) - 0.035
    xmax = max(max(vals), base) + 0.085
    ax.axvline(base, color="#E67E22", ls="--", lw=1.9, zorder=4,
               label=f"惯性基线 AUC = {base:.3f}")
    ax.axvline(0.5, color="gray", ls=":", lw=1.1, alpha=.6, zorder=1)
    ax.text(0.5, len(gids) - 0.32, "随机", ha="center", fontsize=8, color="gray")

    for y_, v, lv in zip(ypos, vals, lrs):
        inside = v > xmax - 0.13
        ax.text(v - 0.007 if inside else v + 0.007, y_ + 0.02, f"{v:.3f}",
                va="center", ha="right" if inside else "left", fontsize=9,
                fontweight="bold" if inside else "normal",
                color="white" if inside else "#2C3E50")
        ax.plot([lv], [y_ - 0.16], marker="D", ms=6.5, color="#1F4E79",
                zorder=6, label="逻辑回归 AUC" if y_ == ypos[0] else None)

    ax.set_yticks(ypos)
    ax.set_yticklabels([SHORT[g] for g in gids], fontsize=9.5)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.7, len(gids) - 0.15)
    ax.set_title(f"{title}\nXGBoost（条）与逻辑回归（菱形）", fontsize=11, pad=8)
    ax.set_xlabel("AUC（14 天时间切分测试集）", fontsize=9.5)
    ax.grid(axis="x", alpha=.28)

h, l = axes[0].get_legend_handles_labels()
h2, l2 = axes[1].get_legend_handles_labels()
seen, hh, ll = set(), [], []
for x_, y_ in list(zip(h, l)) + list(zip(h2, l2)):
    if y_ not in seen:
        seen.add(y_)
        hh.append(x_)
        ll.append(y_)
fig.legend(hh, ll, loc="lower center", ncol=2, fontsize=9.5, frameon=False,
           bbox_to_anchor=(0.5, 0.005))
fig.suptitle("特征组消融：内容属性、当日热度、惯性趋势各自的解释力", fontsize=12.5, y=0.985)
fig.tight_layout(rect=(0, 0.05, 1, 0.96))
fig.savefig(os.path.join(CH, "exp_ablation_auc.png"), dpi=160)
plt.close(fig)
print("已写 charts/exp_ablation_auc.png")

# ================================================================ 图 2
# 只画「剔除 roll7 后的 AUC 下降量」，用零基线差值图。
# 若用堆叠柱 + 截断 y 轴，0.03 的差距会被放大成视觉上的数倍，属误导性图表。
PAIRS = [("G5", "G6", "内容＋惯性\n17 → 16 维"),
         ("G7", "G8", "全部 22 维\n22 → 21 维")]
fig, ax = plt.subplots(figsize=(8.6, 4.9))
x = np.arange(len(PAIRS))
w = 0.32
COL = {"abs": "#7F8C8D", "break": "#1F4E79"}
for i, (key, title) in enumerate(PANELS):
    b = R["scenarios"][key]
    d = [b["groups"][a]["auc_xgb"] - b["groups"][bb]["auc_xgb"] for a, bb, _ in PAIRS]
    pos = x + (i - .5) * w
    ax.bar(pos, d, w, color=COL[key], alpha=.92,
           label="绝对热度口径" if i == 0 else "相对跃升口径")
    for p_, dv, (a, bb, _) in zip(pos, d, PAIRS):
        av, bv = b["groups"][a]["auc_xgb"], b["groups"][bb]["auc_xgb"]
        ax.text(p_, dv + 0.0012, f"−{dv:.3f}", ha="center", fontsize=9.5,
                fontweight="bold", color=COL[key])
        ax.text(p_, dv / 2 if dv > 0.008 else dv + 0.004,
                f"{av:.3f} → {bv:.3f}", ha="center", va="center",
                fontsize=8.2, color="white" if dv > 0.008 else "#444")
ax.axhline(0, color="black", lw=1.2)
ax.set_xticks(x)
ax.set_xticklabels([p[2] for p in PAIRS], fontsize=10)
ax.set_ylim(-0.003, 0.040)
ax.set_ylabel("剔除 roll7_like 后 XGBoost AUC 的下降量（Δ）", fontsize=9.5)
ax.set_title("roll7_like 的分母效应有多大？", fontsize=12, pad=10)
ax.grid(axis="y", alpha=.3)
ax.legend(fontsize=9.5, loc="upper right", frameon=False)
fig.text(0.5, 0.015,
         "柱高 = 模型仅凭「知道标签分母」获得的分数。"
         "绝对口径下几乎为 0（roll7 与标签无关）；跃升口径下约 0.03，说明分母效应存在但不是主力。",
         ha="center", fontsize=8.6, color="#7a7a7a")
fig.tight_layout(rect=(0, 0.055, 1, 1))
fig.savefig(os.path.join(CH, "exp_ablation_roll7.png"), dpi=160)
plt.close(fig)
print("已写 charts/exp_ablation_roll7.png")
