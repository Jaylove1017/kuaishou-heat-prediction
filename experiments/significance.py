# -*- coding: utf-8 -*-
"""显著性检验：把「模型没有增量」从定性判断变成可证伪的结论。

为什么需要这一步
----------------
原文的论证是「模型 AUC 0.9813 < 基线 0.9842，且我知道有 ±0.002 的抖动，
所以是没增量而不是更差」。这是定性判断：0.003 的差到底算不算差，读者无法自己判断。
本脚本给两样东西：
  1. 配对 bootstrap 的 95% 置信区间（按天重采样 —— per-day 口径下「天」才是独立单元，
     按样本重采样会把同一天内高度相关的样本当独立信息，区间会被严重低估）
  2. 多个随机种子的均值 ± 标准差

判读方式
--------
* 差值的 95% CI **跨 0** → 模型与基线在统计上不可区分。这时正确的说法是
  「没有可检测的增量」，而不是「模型更差」。
* CI 整体在 0 以下 → 基线确实更好，可以下「不如一行 SQL」的结论。
* AUC 与 top-k 召回可能给出不同结论（AUC 看全序，top-k 只看头部），
  两个都要报，不能只挑一个。

输出：outputs/exp_significance.json
用法：
    python experiments/significance.py              # 默认 1000 次重采样、5 个种子
    python experiments/significance.py --iters 300  # 快速看一眼
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import features as F

sys.stdout.reconfigure(encoding="utf-8")

KS = 0.20
SEEDS = (42, 1, 2, 3, 4)
OUT = os.path.join(F.PROJ, "outputs")
os.makedirs(OUT, exist_ok=True)


def boot_ci(days, y, p_model, p_naive, k=KS, iters=1000, seed=0):
    """按天重采样，返回 AUC 差与 per-day top-k 召回差的 95% CI"""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    uniq = np.unique(days)
    idx_by_day = {d: np.flatnonzero(days == d) for d in uniq}

    d_auc, d_rec = [], []
    for _ in range(iters):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_day[d] for d in pick])
        yy = y[idx]
        if yy.min() == yy.max():          # 重采样后单一类别，AUC 无定义
            continue
        a = roc_auc_score(yy, p_model[idx])
        b = roc_auc_score(yy, p_naive[idx])
        d_auc.append(a - b)

        # 同一次重采样下算 per-day top-k 召回差（保持两者的相关性，才是配对比较）
        hits_m = hits_n = n_sel = 0
        for d in pick:
            ii = idx_by_day[d]
            n = max(int(len(ii) * k), 1)
            topm = ii[np.argsort(-p_model[ii])[:n]]
            topn = ii[np.argsort(-p_naive[ii])[:n]]
            hits_m += int(y[topm].sum())
            hits_n += int(y[topn].sum())
            n_sel += n
        tot = max(int(yy.sum()), 1)
        d_rec.append((hits_m - hits_n) / tot * 100)

    def ci(v, name):
        v = np.asarray(v)
        lo, hi = np.percentile(v, [2.5, 97.5])
        return dict(name=name, mean=round(float(v.mean()), 4),
                    ci_low=round(float(lo), 4), ci_high=round(float(hi), 4),
                    crosses_zero=bool(lo < 0 < hi), n_boot=int(len(v)))

    return ci(d_auc, "AUC(模型-基线)"), ci(d_rec, "top20%召回差(pp，模型-基线)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1000)
    args = ap.parse_args()

    print("[1] 构造样本 ...")
    ds = F.build_dataset(F.filter_active(F.load_daily_csv()))
    tr, te, split_d = F.split(ds)
    print(f"    样本 {len(ds):,}  train {len(tr):,}  test {len(te):,}  split={split_d}")

    from sklearn.metrics import roc_auc_score
    days = te["d"].values
    result = {"split_day": int(split_d), "test_rows": int(len(te)),
              "test_days": int(len(np.unique(days))), "iters": args.iters,
              "bootstrap_unit": "day（per-day 口径下天是独立单元）", "scenarios": {}}

    SCEN = [
        dict(key="abs", label="label_abs", name="绝对热度（次日进前25%）", naive="like_user_num"),
        dict(key="break", label="label_break", name="相对跃升（≥自身7日均值2倍）", naive="roll7_like"),
    ]

    for sc in SCEN:
        print(f"\n[2] 口径：{sc['name']}")
        ytr, yte = tr[sc["label"]].values, te[sc["label"]].values
        naive = te[sc["naive"]].fillna(-1).values.astype(float)

        per_seed, probs = [], []
        for seed in SEEDS:
            m = F.make_xgb(seed)
            m.fit(tr[F.FEATURES], ytr)
            p = m.predict_proba(te[F.FEATURES])[:, 1]
            probs.append(p)
            auc_m = float(roc_auc_score(yte, p))
            auc_n = float(roc_auc_score(yte, naive))
            per_seed.append(dict(seed=seed, auc_model=round(auc_m, 4), auc_naive=round(auc_n, 4),
                                 diff=round(auc_m - auc_n, 4)))
            print(f"    seed={seed:<3} 模型 {auc_m:.4f}  基线 {auc_n:.4f}  差 {auc_m-auc_n:+.4f}")

        diffs = np.array([r["diff"] for r in per_seed])
        print(f"    跨种子：差 {diffs.mean():+.4f} ± {diffs.std(ddof=1):.4f} "
              f"(min {diffs.min():+.4f} / max {diffs.max():+.4f})")

        print(f"[3] 配对 bootstrap（按天重采样 {args.iters} 次）...")
        ci_auc, ci_rec = boot_ci(days, yte, probs[0], naive, iters=args.iters)
        for c in (ci_auc, ci_rec):
            star = "  ← 跨 0，统计上不可区分" if c["crosses_zero"] else ""
            print(f"    {c['name']:<28} {c['mean']:+.4f} "
                  f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]{star}")
        print(f"    （参照：本次实测基线 AUC {per_seed[0]['auc_naive']:.4f}）")

        result["scenarios"][sc["key"]] = dict(
            name=sc["name"], pos_rate=round(float(yte.mean()) * 100, 2),
            naive_col=sc["naive"], per_seed=per_seed,
            seed_mean=round(float(diffs.mean()), 4),
            seed_std=round(float(diffs.std(ddof=1)), 4),
            bootstrap_auc=ci_auc, bootstrap_top20_recall=ci_rec,
            conclusion=("无可检测增量（差值 95%CI 跨 0）" if ci_auc["crosses_zero"]
                        else ("基线显著更好" if ci_auc["ci_high"] < 0 else "模型显著更好")),
        )
        print(f"    → {result['scenarios'][sc['key']]['conclusion']}")

    with open(os.path.join(OUT, "exp_significance.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\n已写 outputs/exp_significance.json")
    print("提示：README 里凡出现「无增量」，都应指向本文件，而不是靠抖动区间口头判断。")


if __name__ == "__main__":
    main()
