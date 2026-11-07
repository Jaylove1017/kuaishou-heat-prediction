# -*- coding: utf-8 -*-
"""口径审计实验：把 README 里三个「说得很顺但其实没验过」的措辞各自量化一次

被审计的三句话
--------------
1. 「预测次日点赞」
   实际标签用的是 next_d = 该视频**下一次出现**的日期，不一定是 d+1。
   本实验给出 gap_days 的真实分布，以及只保留 gap_days == 1 后指标掉多少。
2. 「冷启动也能预测」
   22 维里有 lag1/2/3，默认 dropna 会把 n_prior_days ≤ 2 的样本全部删掉。
   本实验给出被删掉多少行、正样本率多少，以及保留后（XGBoost 原生吃 NaN）
   冷启动段到底能不能预测。
3. 「按时间切分，没有穿越」
   默认按**特征日 d** 切分，于是训练集里存在 d ≤ split 但 next_d > split 的样本，
   它们的标签落在测试窗口内。本实验改为按 next_d 切分，量化这一点泄漏有多大。

四种配置
--------
  A 默认（历史口径）      gap 不限 + dropna + 按 d 切分
  B 严格次日              gap ≤ 1  + dropna + 按 d 切分
  C 保留冷启动            gap 不限 + 不 dropna + 按 d 切分
  D 严格次日 + 按目标日切  gap ≤ 1  + 不 dropna + 按 next_d 切分   ← 最干净

输出：outputs/exp_strict.json
用法：python experiments/strict_eval.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import features as F

sys.stdout.reconfigure(encoding="utf-8")

PROJ = F.PROJ
OUT = os.path.join(PROJ, "outputs")
os.makedirs(OUT, exist_ok=True)

CONFIGS = [
    ("A", "默认（历史口径）", dict(gap_max=None, dropna=True, split_on="d")),
    ("B", "严格次日", dict(gap_max=1, dropna=True, split_on="d")),
    ("C", "保留冷启动", dict(gap_max=None, dropna=False, split_on="d")),
    ("D", "严格次日+按目标日切分", dict(gap_max=1, dropna=False, split_on="next_d")),
]

SEGMENTS = [(-1, 0, "首次活跃(0天)"), (0, 2, "1-2天"), (2, 6, "3-6天"), (6, 10 ** 9, "7天以上")]


def segment_report(p, y, days, n_prior):
    """按历史活跃天数分段，报每段的样本量、正样本率与段内 top20% 召回
    （模型 vs 按当日点赞排序）。段内圈选是因为业务上新视频只会与同段竞品比。"""
    n_prior = np.asarray(n_prior)
    out = {}
    for lo, hi, name in SEGMENTS:
        m = (n_prior > lo) & (n_prior <= hi)
        n = int(m.sum())
        if n == 0:
            out[name] = dict(n=0, note="该口径下此段无样本")
            continue
        e = F.eval_per_day(p[m], y[m], days[m], 0.20)
        out[name] = dict(n=n, pos=round(float(y[m].mean()) * 100, 2),
                         top20_recall=round(e["recall"], 2))
    return out


def main():
    from sklearn.metrics import roc_auc_score

    print("[1] 读取数据 ...")
    daily = F.filter_active(F.load_daily_csv())

    result = {"data_rows_active": int(len(daily)), "configs": {}}

    # ---- 先报目标间隔分布，这是「次日」这个说法的全部依据 ----
    # 用 lead_days = next_d - d（向前），不是 gap_days（向后，距上次观测）。
    base = F.build_dataset(daily, dropna=False)
    lead = base["lead_days"]
    back = base["gap_days"]
    gap_dist = {
        "n_samples": int(len(lead)),
        "share_gap_le1": round(float((lead <= 1).mean()) * 100, 2),
        "share_gap_le7": round(float((lead <= 7).mean()) * 100, 2),
        "median": float(lead.median()), "p90": float(lead.quantile(0.90)),
        "max": float(lead.max()),
        "backward_gap_median": float(back.median()),
        "backward_gap_share_le1": round(float((back <= 1).mean()) * 100, 2),
    }
    result["gap_days_distribution"] = gap_dist
    print(f"\n[2] 目标间隔 next_d - d（向前）：中位数 {gap_dist['median']:.0f} 天，"
          f"P90 {gap_dist['p90']:.0f} 天，最大 {gap_dist['max']:.0f} 天")
    print(f"    真正「次日」(lead==1) 的样本占比 {gap_dist['share_gap_le1']:.2f}%，"
          f"lead≤7 占 {gap_dist['share_gap_le7']:.2f}%")
    print(f"    （对照：向后间隔 gap_days 中位数 {gap_dist['backward_gap_median']:.0f} 天，"
          f"≤1 占 {gap_dist['backward_gap_share_le1']:.2f}% —— 两者含义不同，不可混用）")

    # ---- 冷启动样本被 dropna 删掉多少 ----
    all_rows = len(base)
    kept = len(F.build_dataset(daily, dropna=True))
    cold = base["n_prior_days"] <= 2
    result["cold_start_loss"] = {
        "rows_with_next_day": int(all_rows),
        "rows_after_dropna": int(kept),
        "dropped": int(all_rows - kept),
        "dropped_share_pct": round((all_rows - kept) / all_rows * 100, 2),
        "cold_rows_total": int(cold.sum()),
        "cold_share_of_all_pct": round(float(cold.mean()) * 100, 2),
        "cold_pos_rate": round(float(base.loc[cold, "label_abs"].mean()) * 100, 2),
        "warm_pos_rate": round(float(base.loc[~cold, "label_abs"].mean()) * 100, 2),
    }
    print(f"\n[3] 冷启动：n_prior_days≤2 的样本 {cold.sum():,} 行 "
          f"（占 {cold.mean()*100:.1f}%），默认 dropna 后从 {all_rows:,} 掉到 {kept:,} 行")
    print(f"    冷启动段正样本率 {result['cold_start_loss']['cold_pos_rate']:.2f}%，"
          f"老视频段 {result['cold_start_loss']['warm_pos_rate']:.2f}%")

    # ---- 四种配置各跑一遍 ----
    for cid, name, opt in CONFIGS:
        print(f"\n[4-{cid}] {name}  gap_max={opt['gap_max']} dropna={opt['dropna']} "
              f"split_on={opt['split_on']}")
        ds = F.build_dataset(daily, gap_max=opt["gap_max"], dropna=opt["dropna"])
        tr, te, split_d = F.split(ds, on=opt["split_on"])
        if te.empty or tr.empty:
            print("    跳过：切分后无样本")
            continue

        ytr, yte = tr["label_abs"].values, te["label_abs"].values
        m = F.make_xgb(F.SEED)
        m.fit(tr[F.FEATURES], ytr)
        p = m.predict_proba(te[F.FEATURES])[:, 1]
        naive = te["like_user_num"].fillna(-1).values.astype(float)
        days = te["d"].values

        auc_m, auc_n = float(roc_auc_score(yte, p)), float(roc_auc_score(yte, naive))
        e20m = F.eval_per_day(p, yte, days, 0.20)
        e20n = F.eval_per_day(naive, yte, days, 0.20)
        seg = segment_report(p, yte, days, te["n_prior_days"].values)

        leak = 0
        if opt["split_on"] == "d":
            leak = int(((tr["d"] <= split_d) & (tr["next_d"] > split_d)).sum())

        print(f"    样本 {len(ds):,}  train {len(tr):,}  test {len(te):,}  pos {yte.mean()*100:.2f}%")
        print(f"    AUC 模型 {auc_m:.4f} / 基线 {auc_n:.4f}  (差 {auc_m-auc_n:+.4f})")
        print(f"    top20% 召回 模型 {e20m['recall']:.2f}% / 基线 {e20n['recall']:.2f}%")
        if leak:
            print(f"    训练集中标签落在测试窗口的样本 {leak:,} 行")
        print("    分段 top20% 召回：" + "  ".join(
            f"{k}={v.get('top20_recall', '—')}%(n={v['n']:,})" for k, v in seg.items()))

        result["configs"][cid] = dict(
            name=name, gap_max=opt["gap_max"], dropna=opt["dropna"], split_on=opt["split_on"],
            n_samples=int(len(ds)), n_train=int(len(tr)), n_test=int(len(te)),
            split_day=int(split_d), pos_rate=round(float(yte.mean()) * 100, 2),
            auc_model=round(auc_m, 4), auc_naive=round(auc_n, 4),
            auc_diff=round(auc_m - auc_n, 4),
            top20_model=round(e20m["recall"], 2), top20_naive=round(e20n["recall"], 2),
            top20_diff=round(e20m["recall"] - e20n["recall"], 2),
            train_label_in_test_window=leak, by_history=seg)

    with open(os.path.join(OUT, "exp_strict.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\n已写 outputs/exp_strict.json")


if __name__ == "__main__":
    main()
