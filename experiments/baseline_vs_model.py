# -*- coding: utf-8 -*-
"""基线对照（**旧口径 · 仅作追溯**）：同一测试集上比「模型 vs 按当日点赞排序」，
并按历史活跃长度做增益拆解。

⚠ 引用前必读
------------
本脚本的 top-k 用**全局混排**（14 天测试样本放在一起取 top-k%），
业务上没有这个动作 —— 每天收盘只圈一次明天的候选。
因此它算出的 top20 召回是 **76.17% / 76.39%**，
与 experiments/eval_upgrade.py 的 per-day 口径 **77.01% / 77.22%** 不是同一组数字。

README 正文与总表一律引用 per-day 口径（eval_upgrade.py），**不要混引本文件的数字**。
保留本脚本的价值只有两个：
  1. 它是「先立基线」这一步的原始证据（时间上早于口径升级）；
  2. 它按 n_prior_days 分段拆增益 —— 这一段 eval_upgrade.py 没做。

关于分段：默认口径按 FEATURES 全集 dropna，而 lag1/2/3 的存在使
n_prior_days ≤ 2 的样本必然被删，所以「首次活跃(0天)」「1-2天」两段恒为空。
这不是 bug，是「冷启动样本不进模型」这个事实的体现；现在把它显式打印为 0，
而不是静默跳过。想看冷启动表现请用 model_analysis.py --keep-na。
"""
import sys
import os
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import features as F

sys.stdout.reconfigure(encoding="utf-8")
PROJ = F.PROJ
SPLIT_TAIL_DAYS, TOP_Q = F.SPLIT_TAIL_DAYS, F.TOP_Q

df = F.filter_active(F.load_daily_csv())
ds = F.build_dataset(df)
FEATURES = F.FEATURES

tr, te, split_d = F.split(ds)
print(f"样本 {len(ds):,}  train {len(tr):,}  test {len(te):,}  split={split_d}"
      f"  （全局混排口径；正式结论见 eval_upgrade.py）")

from sklearn.metrics import roc_auc_score

Xtr, ytr = tr[FEATURES], tr["label_abs"].values
Xte, yte = te[FEATURES], te["label_abs"].values
lr = F.make_lr()
lr.fit(Xtr, ytr)
p_lr = lr.predict_proba(Xte)[:, 1]
xm = F.make_xgb()
xm.fit(Xtr, ytr)
p_xgb = xm.predict_proba(Xte)[:, 1]
print(f"[复现校验] LR AUC={roc_auc_score(yte,p_lr):.4f}  XGB AUC={roc_auc_score(yte,p_xgb):.4f}")

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
    n = int(m.sum())
    if n == 0:
        # 显式报 0：这四段里前两段按当前口径必然为空（lag 缺失被 dropna）
        print(f"{name:<14}{0:>9,}   ——（该段被 dropna 删除，冷启动样本不进模型）")
        seg_res[name] = dict(n=0, note="当前口径下为空：lag1-3 缺失，被 dropna 删除")
        continue
    y = yte[m.values]
    a, b = rec(p_xgb[m.values], y, .2), rec(naive[m.values], y, .2)
    seg_res[name] = dict(n=n, pos=round(float(y.mean()) * 100, 2),
                         model=round(a, 2), naive=round(b, 2), lift=round(a - b, 2))
    print(f"{name:<14}{n:>9,}{y.mean()*100:>8.2f}%{a:>8.2f}%{b:>8.2f}%{a-b:>+8.2f}pp")

imp = sorted(zip(FEATURES, xm.feature_importances_), key=lambda t: -t[1])
print("\n前 6 特征(gain):", [(f, round(float(v), 4)) for f, v in imp[:6]])

with open(os.path.join(PROJ, "outputs", "baseline_vs_model.json"), "w", encoding="utf-8") as f:
    json.dump(dict(eval_mode="global（旧口径，勿与 eval_upgrade 的 per-day 混引）",
                   test_rows=int(len(te)), split_day=int(split_d),
                   auc_model=round(float(roc_auc_score(yte, p_xgb)), 4),
                   auc_naive_today=round(float(roc_auc_score(yte, naive)), 4),
                   auc_naive_lag1=round(float(roc_auc_score(yte, te["lag1_like"].fillna(-1).values)), 4),
                   topk={str(k): v for k, v in res.items()}, by_history=seg_res),
              f, ensure_ascii=False, indent=2)
print("\n已写 outputs/baseline_vs_model.json")
