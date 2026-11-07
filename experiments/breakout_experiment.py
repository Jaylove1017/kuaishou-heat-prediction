# -*- coding: utf-8 -*-
"""延伸实验：把标签从「绝对高潜」换成「相对自身的跃升(breakout)」，
检验模型在惯性解释不了的地方是否有真增量 —— 用于回答「为什么不直接按当日点赞排序」。

⚠ 本脚本是【早期版本】，评估只用全局混排口径（把 14 天测试集混在一起取 top-k%），
   真实业务里没有这个动作。**正式结论请引用 experiments/eval_upgrade.py 的
   per-day 口径**，本脚本保留是为了追溯「先发现现象、后升级口径」这个过程。

本脚本额外补了一组排序线，用来检验 4.3 节那个「AUC 0.43 = 均值回归」的说法：
   按 roll7 排序（分母越大越难跃升）→ 预期 AUC < 0.5
   按 1/roll7 排序                    → 预期 AUC > 0.5
   按 当日点赞/roll7（归一化动量）排序 → 若均值回归成立，应明显强于前两条
只有后两条也成立，「均值回归」才是被数据支持的解释，而不是构造性的恒等式。
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

df = F.filter_active(F.load_daily_csv())
ds = F.build_dataset(df)                 # 统一按 22 维全集 dropna
FEATURES = F.FEATURES

# label_abs / label_break 由 features.build_dataset 一并产出（定义见该文件头部）
tr, te, split_d = F.split(ds)
print(f"样本 {len(ds):,}  train {len(tr):,}  test {len(te):,}  split={split_d}")

from sklearn.metrics import roc_auc_score, average_precision_score


def rec(s, y, k):
    order = np.argsort(-np.asarray(pd.Series(s).fillna(-1).values, float))
    top = order[: int(len(y) * k)]
    return y[top].sum() / max(y.sum(), 1) * 100


def run(labelcol, naive_col, naive_name, feats=FEATURES):
    Xtr, ytr = tr[feats], tr[labelcol].values
    Xte, yte = te[feats], te[labelcol].values
    if yte.sum() == 0:
        print("无正样本")
        return None
    m = F.make_xgb()
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    naive = te[naive_col].fillna(-1).values
    print(f"\n=== 标签：{labelcol}  (正样本率 test {yte.mean()*100:.2f}%) ===")
    print(f"  模型 XGB      AUC={roc_auc_score(yte,p):.4f}  AP={average_precision_score(yte,p):.4f}  "
          f"top20%召回={rec(p,yte,.2):.2f}%（全局混排口径）")
    print(f"  朴素 {naive_name:<12} AUC={roc_auc_score(yte,naive):.4f} "
          f" top20%召回={rec(naive,yte,.2):.2f}%")
    out = dict(pos_rate=round(float(yte.mean()) * 100, 2),
               auc_model=round(float(roc_auc_score(yte, p)), 4),
               auc_naive=round(float(roc_auc_score(yte, naive)), 4),
               top20_model=round(rec(p, yte, .2), 2), top20_naive=round(rec(naive, yte, .2), 2))
    for k in (.05, .10, .20, .30):
        out[f"top{int(k*100)}_model"] = round(rec(p, yte, k), 2)
        out[f"top{int(k*100)}_naive"] = round(rec(naive, yte, k), 2)
    print("  top-k 对照:", {f"top{int(k*100)}%": f"模型{rec(p,yte,k):.1f}/朴素{rec(naive,yte,k):.1f}"
                            for k in (.05, .10, .20, .30)})
    return out


res = {}
res["绝对高潜(原口径)"] = run("label_abs", "like_user_num", "当日点赞")
res["相对跃升(breakout)"] = run("label_break", "roll7_like", "近7日均值")

# 跃升口径下，只用历史特征（去掉当日 like_user_num/play_user_num 等当日热度），
# 看纯「内容属性+趋势」能预测多少 —— 这才是模型不可替代的部分
feats_nocur = [f for f in FEATURES if f not in
               ("like_user_num", "play_user_num", "complete_play_cnt", "like_rate", "complete_rate")]
print("\n[跃升口径] 去掉所有'当日热度'特征，只用内容属性+滞后/滚动趋势：")
print("  参与特征:", feats_nocur)
res["跃升+去当日热度"] = run("label_break", "roll7_like", "近7日均值(全特征基线)", feats=feats_nocur)

# ---- 方向检验：AUC 0.43 到底是均值回归，还是构造性反向？ ----
print("\n[方向检验] 跃升标签下，用三条不同的排序线看 AUC：")
y = te["label_break"].values
r7 = te["roll7_like"].replace(0, np.nan)
lines = {
    "按 roll7 排序（分母越大越难跃升）": r7,
    "按 1/roll7 排序（越冷越优先）": 1.0 / r7,
    "按 当日点赞/roll7 归一化动量排序": te["like_user_num"] / r7,
}
dirs = {}
for name, s in lines.items():
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    s = s.fillna(s.median())
    auc = float(roc_auc_score(y, s.values))
    dirs[name] = round(auc, 4)
    print(f"  {name:<32} AUC={auc:.4f}  top20%召回={rec(s.values, y, .2):.2f}%")
print("  解读：第一行 <0.5 是标签定义（next_like ≥ 2×roll7）的构造性结果，不算发现；")
print("        第二、三行才是「均值回归是否真的存在」的证据。")
res["方向检验"] = dirs

with open(os.path.join(PROJ, "outputs", "breakout_experiment.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print("\n已写 outputs/breakout_experiment.json")
