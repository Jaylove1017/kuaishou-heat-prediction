# -*- coding: utf-8 -*-
"""特征组消融实验（论文第 6 章：内容 vs 热度 vs 惯性的贡献分解）

背景
----
原有两组消融是"手写剔除"：
  22 维（全部）            AUC 0.8822（跃升口径）
  17 维（去 5 个当日热度）  AUC 0.8087
这两组都保留了 12 维惯性特征，因此无法回答三个问题：
  1. 只靠"内容属性"能解释多少？（从未跑过）
  2. "内容 + 当日热度"能解释多少？（从未跑过）
  3. 跃升口径下 roll7_like 既是特征又是标签分母，模型有多少分是靠"记住分母"拿到的？

本脚本按语义把 22 维拆成三组，做 8 组消融 × 2 种标签 × 2 个模型，
用来分离上述三个来源。

特征分组（22 维）
----------------
  内容属性 CONTENT   5 : dur_log, is_ad, up_type_idx, tag_idx, dow
  当日热度 HEAT      5 : play_user_num, like_user_num, complete_play_cnt, like_rate, complete_rate
  惯性趋势 INERTIA  12 : lag1/2/3_like, lag1/2/3_playuser, lag1_likerate,
                         roll7_like, roll7_playuser, roll7_complete, n_prior_days, gap_days

消融组（8 组）
-------------
  G1 仅内容              5
  G2 仅当日热度           5
  G3 仅惯性              12
  G4 内容+当日           10
  G5 内容+惯性           17  ← 等价于旧脚本的 feats_nocur
  G6 内容+惯性−roll7     16  ← 关键新增：剔掉标签分母
  G7 全部                22  ← 等价于旧 FEATURES
  G8 全部−roll7          21  ← 关键新增：剔掉标签分母

一致性保证
----------
  - 样本集固定：统一用 22 维全集做 dropna，各组共用同一批样本（否则不可比）
  - 时间切分 / 标签定义 / 超参 与 eval_upgrade.py 完全一致
  - 评估口径统一用 per-day top20%（业务正确口径），同时保留 AUC / AP

输出
----
  outputs/exp_ablation.json
  charts/exp_ablation_auc.png      分组 AUC 对比（两面板）
  charts/exp_ablation_roll7.png    roll7 分母效应专项对比
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
SEED = F.SEED
OUT = os.path.join(PROJ, "outputs")
CH = os.path.join(PROJ, "charts")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CH, exist_ok=True)

# ---------------------------------------------------------------- 数据与特征
print("[1] 读取 CSV 并构造特征 ...")
df = F.filter_active(F.load_daily_csv())
# 统一按 22 维全集 dropna：各组共用同一批样本，否则组间不可比
ds = F.build_dataset(df)
ds = ds.sort_values(["d", "video_id"]).reset_index(drop=True)

# ---------------------------------------------------------------- 特征分组
# 分组定义已上移到 src/features.py（与 model_analysis / significance 共用）
CONTENT, HEAT, INERTIA = F.CONTENT, F.HEAT, F.INERTIA
ALL22 = F.FEATURES
assert len(CONTENT) == 5 and len(HEAT) == 5 and len(INERTIA) == 12
assert len(set(ALL22)) == 22, "特征重复"

INERTIA_NOR7 = [f for f in INERTIA if f != "roll7_like"]
ALL_NOR7 = [f for f in ALL22 if f != "roll7_like"]

GROUPS = [
    ("G1", "仅内容属性", CONTENT),
    ("G2", "仅当日热度", HEAT),
    ("G3", "仅惯性趋势", INERTIA),
    ("G4", "内容＋当日热度", CONTENT + HEAT),
    ("G5", "内容＋惯性", CONTENT + INERTIA),
    ("G6", "内容＋惯性−roll7", CONTENT + INERTIA_NOR7),
    ("G7", "全部 22 维", ALL22),
    ("G8", "全部−roll7", ALL_NOR7),
]

tr, te, split_d = F.split(ds)
days_te = te["d"].values
print(f"    样本 {len(ds):,}  train {len(tr):,}  test {len(te):,}  split={split_d}")
print(f"    测试集 {te['d'].nunique()} 天，每日样本 {te.groupby('d').size().min()}–{te.groupby('d').size().max()}")


# ---------------------------------------------------------------- 评估函数（与 eval_upgrade.py 一致）
def eval_per_day(scores, y, days, k=0.20):
    d = F.eval_per_day(scores, y, days, k)
    return dict(recall=d["recall"], precision=d["precision"])


from sklearn.metrics import roc_auc_score, average_precision_score


SCENARIOS = [
    dict(key="abs", label="label_abs", name="绝对热度（次日进前25%）", naive_col="like_user_num"),
    dict(key="break", label="label_break", name="相对跃升（≥自身7日均值2倍）", naive_col="roll7_like"),
]

result = {
    "split_day": int(split_d), "test_rows": int(len(te)),
    "test_days": int(te["d"].nunique()), "seed": SEED,
    "feature_groups": {"CONTENT": CONTENT, "HEAT": HEAT, "INERTIA": INERTIA},
    "groups": [{"id": gid, "name": nm, "n_feat": len(f), "features": f} for gid, nm, f in GROUPS],
    "scenarios": {},
}


def fit_eval(feats, ytr, yte):
    Xtr, Xte = tr[feats], te[feats]
    lr = F.make_lr(SEED)
    lr.fit(Xtr, ytr)
    p_lr = lr.predict_proba(Xte)[:, 1]
    xm = F.make_xgb(SEED)
    xm.fit(Xtr, ytr)
    p_xgb = xm.predict_proba(Xte)[:, 1]
    return p_lr, p_xgb


for sc in SCENARIOS:
    ytr = tr[sc["label"]].values
    yte = te[sc["label"]].values
    naive = te[sc["naive_col"]].fillna(-1).values.astype(float)
    nv = eval_per_day(naive, yte, days_te)
    print(f"\n[2] 口径：{sc['name']}")
    print(f"    正样本率 test={yte.mean()*100:.2f}%   惯性基线 AUC={roc_auc_score(yte, naive):.4f} "
          f"AP={average_precision_score(yte, naive):.4f} top20% R={nv['recall']:.2f}% P={nv['precision']:.2f}%")

    block = dict(name=sc["name"], pos_rate=round(float(yte.mean()) * 100, 2),
                 baseline=dict(col=sc["naive_col"],
                               auc=round(float(roc_auc_score(yte, naive)), 4),
                               ap=round(float(average_precision_score(yte, naive)), 4),
                               top20=nv),
                 groups={})
    print(f"    {'组':<22}{'维':>4}{'AUC-LR':>10}{'AUC-XGB':>10}{'AP-XGB':>9}"
          f"{'R@20':>8}{'P@20':>8}{'vs基线':>9}")
    for gid, gname, feats in GROUPS:
        p_lr, p_xgb = fit_eval(feats, ytr, yte)
        t20 = eval_per_day(p_xgb, yte, days_te)
        rec = dict(
            id=gid, name=gname, n_feat=len(feats),
            auc_lr=round(float(roc_auc_score(yte, p_lr)), 4),
            auc_xgb=round(float(roc_auc_score(yte, p_xgb)), 4),
            ap_lr=round(float(average_precision_score(yte, p_lr)), 4),
            ap_xgb=round(float(average_precision_score(yte, p_xgb)), 4),
            top20_xgb=t20,
            lift_recall_vs_naive=round(t20["recall"] - nv["recall"], 2),
        )
        block["groups"][gid] = rec
        print(f"    {gname:<20}{len(feats):>4}{rec['auc_lr']:>10.4f}{rec['auc_xgb']:>10.4f}"
              f"{rec['ap_xgb']:>9.4f}{t20['recall']:>7.2f}%{t20['precision']:>7.2f}%"
              f"{rec['lift_recall_vs_naive']:>+8.2f}pp")
    result["scenarios"][sc["key"]] = block

with open(os.path.join(OUT, "exp_ablation.json"), "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print("\n[3] 已写 outputs/exp_ablation.json")

# ---------------------------------------------------------------- 图 1：分组 AUC
print("[4] 画图 ...")
short = ["G1 仅内容", "G2 仅热度", "G3 仅惯性", "G4 内容+热度",
         "G5 内容+惯性", "G6 内容+惯性\n−roll7", "G7 全部22维", "G8 全部\n−roll7"]
gids = [g[0] for g in GROUPS]

fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2))
for ax, key, title in zip(axes, ("abs", "break"),
                          ("绝对热度口径（次日进前 25%）", "相对跃升口径（≥自身 7 日均值 2 倍）")):
    b = result["scenarios"][key]
    vals = [b["groups"][g]["auc_xgb"] for g in gids]
    lrs = [b["groups"][g]["auc_lr"] for g in gids]
    ypos = np.arange(len(gids))[::-1]
    colors = ["#C0392B" if g in ("G7", "G8", "G5", "G6") else "#7F8C8D" for g in gids]
    ax.barh(ypos, vals, color=colors, alpha=.85, height=.62)
    ax.plot(lrs, ypos, "D", color="#1F4E79", ms=6, label="逻辑回归 AUC")
    ax.axvline(b["baseline"]["auc"], color="#E67E22", ls="--", lw=1.8,
               label=f"惯性基线 AUC={b['baseline']['auc']:.3f}")
    ax.axvline(0.5, color="gray", ls=":", lw=1.2, alpha=.7)
    for y_, v in zip(ypos, vals):
        ax.text(v + 0.006, y_, f"{v:.3f}", va="center", fontsize=8.5, color="#2C3E50")
    ax.set_yticks(ypos)
    ax.set_yticklabels(short, fontsize=9)
    lo = min(min(vals), min(lrs), b["baseline"]["auc"]) - 0.05
    ax.set_xlim(max(lo, 0.40), 1.0)
    ax.set(title=title + "：各组特征 AUC", xlabel="AUC（测试集）")
    ax.grid(axis="x", alpha=.3)
    ax.legend(fontsize=8.5, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(CH, "exp_ablation_auc.png"), dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- 图 2：roll7 分母效应
fig, ax = plt.subplots(figsize=(7.6, 5.0))
pairs = [("G5", "G6", "内容＋惯性"), ("G7", "G8", "全部 22 维")]
x = np.array([0, 1])
w = 0.34
for i, key in enumerate(("abs", "break")):
    b = result["scenarios"][key]
    with_r7 = [b["groups"][p[0]]["auc_xgb"] for p in pairs]
    no_r7 = [b["groups"][p[1]]["auc_xgb"] for p in pairs]
    ax.bar(x + (i - .5) * w, with_r7, w, color="#1F4E79" if i else "#85929E",
           alpha=.9, label=("跃升口径 含 roll7" if i else "绝对口径 含 roll7"))
    ax.bar(x + (i - .5) * w, [n - w_ for n, w_ in zip(no_r7, with_r7)], w,
           bottom=with_r7, color="#C0392B", alpha=.85,
           label=("跃升口径 剔除 roll7" if i else "绝对口径 剔除 roll7"))
    for xx, a, bb in zip(x + (i - .5) * w, with_r7, no_r7):
        ax.text(xx, max(a, bb) + 0.012, f"{a:.3f}\n→{bb:.3f}", ha="center",
                fontsize=8.2, color="#2C3E50")
        if bb < a:
            ax.text(xx, (a + bb) / 2, f"−{a-bb:.3f}", ha="center", va="center",
                    fontsize=8.5, color="white", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"{p[2]}\n{p[0]} → {p[1]}" for p in pairs], fontsize=9.5)
ax.set_ylim(0.4, 1.0)
ax.set_ylabel("XGBoost AUC")
ax.set_title("roll7_like 的分母效应：剔除它之后 AUC 掉多少", fontsize=11)
ax.grid(axis="y", alpha=.3)
h, l = ax.get_legend_handles_labels()
ax.legend(h[:2], l[:2], fontsize=8.5, loc="upper left")
ax.text(1.0, 0.43, "红段高度 = 模型仅凭「记住标签分母」获得的能力",
        ha="center", fontsize=8.5, color="#7a7a7a")
fig.tight_layout()
fig.savefig(os.path.join(CH, "exp_ablation_roll7.png"), dpi=160)
plt.close(fig)

print("    已写 charts/exp_ablation_auc.png, charts/exp_ablation_roll7.png")
print("\n完成。")
