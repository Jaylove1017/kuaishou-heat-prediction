# -*- coding: utf-8 -*-
"""播放日志特征实验：**1229 万条 play_log 一行没进过模型**，这是本项目最大的浪费

问题
----
`video_daily` 只有「当天有多少人播、多少人赞」这种**总量**，
没有任何刻画「这些人看得怎么样」的**分布**。
模型因此学到的是趋势形态，拿不到内容质量信号 —— 这也是
「模型打不过一行 SQL」最可能的真实原因：它压根没看到比 SQL 更多的信息。

`play_log` 里有最经典的两类内容质量信号：
  1. 完播率分布（p50 / p90、看完率、快看率）—— 内容好不好，用户身体很诚实
  2. 重复观看率（watch_ratio > 1.5）—— 愿意回看是最强的正向信号

本实验回答两个问题
------------------
Q1 把 play_log 的分布特征加进 22 维，模型能不能赢过「按当日点赞排序」？
Q2 如果能，增量是来自内容质量，还是又一个「知道分母」的构造性优势？

对 Q2 的处理：同时报 AUC（全序）与按天 top-k（业务动作），
并给出「相对自身 7 日均值的跃升」这个已经把分母归一化过的口径 ——
在这个口径上，基线 AUC 只有 0.42，任何真实的内容质量信号都会显形。

泄漏说明（重要）
----------------
所有 play_log 特征只用**当日 d** 的播放记录聚合，标签是**次日**点赞，
预测时点在 d 日收盘，全部信息都可得，无穿越。
watch_ratio 由 play_duration / video_duration 得到，两条都用视频自身时长，
不涉及任何未来信息。

输出
----
  outputs/exp_playlog.json

用法：
    python experiments/playlog_features.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import features as F

sys.stdout.reconfigure(encoding="utf-8")

PROJ = F.PROJ
OUT = os.path.join(PROJ, "outputs")
os.makedirs(OUT, exist_ok=True)

PLAY_COLS = ["user_id", "video_id", "play_duration_ms", "video_duration_ms",
             "d", "hour", "watch_ratio", "is_valid"]

# 由 play_log 聚合出的分布特征
PLAY_FEATURES = ["pl_log_n", "pl_uniq_ratio", "pl_wr_p50", "pl_wr_p90",
                 "pl_fast_rate", "pl_done_rate", "pl_rep_rate"]
# 由已有列直接构造的显式比率（「加速度」）
RATIO_FEATURES = ["accel_play", "accel_like", "ctr_like"]


def load_play_log(path=None):
    """读 1229 万行播放日志。play_log.csv 无表头，列为 (user, video, 时长, ...)。"""
    path = path or os.path.join(PROJ, "data", "clean", "play_log.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到 {path}\n请先按 data/README.md 下载 big_matrix.csv.gz 并运行 src/load_data.py")
    return pd.read_csv(
        path, header=None, names=PLAY_COLS,
        dtype={"user_id": "int32", "video_id": "int32", "play_duration_ms": "int64",
               "video_duration_ms": "int64", "d": "int32", "hour": "int8",
               "watch_ratio": "float32", "is_valid": "int8"})


def aggregate_play_features(pl):
    """按 (video_id, d) 聚合出内容质量分布特征。

    只用当日记录 —— 预测时点在当日收盘，不构成穿越。
    """
    pl = pl[(pl["is_valid"] == 1) & (pl["d"] >= F.MIN_DAY)]
    wr = pl["watch_ratio"].clip(0, 10)          # 与 load_data 的异常流剔除口径一致
    pl = pl.assign(wr=wr)

    g = pl.groupby(["video_id", "d"], sort=False)
    feat = g.agg(
        pl_n=("wr", "size"),
        pl_users=("user_id", "nunique"),
        pl_wr_p50=("wr", lambda s: s.quantile(0.50)),
        pl_wr_p90=("wr", lambda s: s.quantile(0.90)),
        # 快看率：没看到 20% 就划走，内容留不住人
        pl_fast_rate=("wr", lambda s: float((s < 0.20).mean())),
        # 看完率：看到 90% 以上
        pl_done_rate=("wr", lambda s: float((s >= 0.90).mean())),
        # 重复观看率：看超 1.5 遍（KuaiRec 允许重复播放，watch_ratio 可 > 1）
        pl_rep_rate=("wr", lambda s: float((s > 1.50).mean())),
    ).reset_index()

    feat["pl_log_n"] = np.log1p(feat["pl_n"])
    feat["pl_uniq_ratio"] = feat["pl_users"] / feat["pl_n"].clip(lower=1)
    return feat[["video_id", "d"] + PLAY_FEATURES]


def add_ratio_features(ds):
    """显式加速度：今天的量相对自身近 7 日均值的倍数。

    原来的 22 维里 play_user_num 与 roll7_playuser 是两个独立特征，
    树模型需要自己学出比值；这里直接把比值给它，看是否等价于「模型能不能自己学会」。
    """
    ds = ds.copy()
    ds["accel_play"] = ds["play_user_num"] / ds["roll7_playuser"].replace(0, np.nan)
    ds["accel_like"] = ds["like_user_num"] / ds["roll7_like"].replace(0, np.nan)
    ds["ctr_like"] = ds["like_user_num"] / ds["play_user_num"].replace(0, np.nan)
    return ds


def main():
    from sklearn.metrics import roc_auc_score

    print("[1] 读取数据 ...")
    daily = F.filter_active(F.load_daily_csv())
    pl = load_play_log()
    print(f"    video_daily {len(daily):,} 行 | play_log {len(pl):,} 行")

    print("[2] 聚合 play_log 分布特征 ...")
    pf = aggregate_play_features(pl)
    print(f"    聚合到 {len(pf):,} 个 (视频,日) 组合，覆盖 "
          f"{pf['video_id'].nunique():,} 个视频")

    print("[3] 构造 22 维基线样本 ...")
    ds = F.build_dataset(daily)
    ds = add_ratio_features(ds)

    # 关键设计：**不取交集**。
    # play_log 来自 big_matrix（7176 个采样用户），只能覆盖约 20% 的「视频×日」。
    # 若只在这个交集上评估，样本会被系统性偏向热门视频 —— 实测交集内正样本率
    # 从 24.92% 涨到 58.37%，而正样本率 58% 时「圈 top 20%」的召回上限只有 34%，
    # 指标直接被机械封顶，任何增益都看不出来。所以这里保留全样本，
    # 未覆盖的行让 play 特征为 NaN（XGBoost 原生支持缺失值，会走默认分支），
    # 这样四组面对的是**同一批 27.9 万样本、同一个 24.92% 正样本率**，
    # 唯一的变量就是「给不给模型这批播放日志信息」。
    ds = ds.merge(pf, on=["video_id", "d"], how="left")
    covered = ds[PLAY_FEATURES].notna().all(axis=1)
    print(f"    全样本 {len(ds):,}，其中 play_log 可覆盖 {int(covered.sum()):,} "
          f"（{covered.mean()*100:.1f}%），未覆盖行保留 NaN 交给 XGBoost")

    tr, te, split_d = F.split(ds)
    print(f"    train {len(tr):,}  test {len(te):,}  split={split_d}")
    days_te = te["d"].values

    GROUPS = [
        ("G0", "22 维基线", F.FEATURES),
        ("G1", "22 维 + 播放分布", F.FEATURES + PLAY_FEATURES),
        ("G2", "22 维 + 显式比率", F.FEATURES + RATIO_FEATURES),
        ("G3", "全部", F.FEATURES + PLAY_FEATURES + RATIO_FEATURES),
    ]

    result = {"split_day": int(split_d), "test_rows": int(len(te)),
              "test_days": int(te["d"].nunique()),
              "play_coverage_pct": round(float(covered.mean()) * 100, 1),
              "coverage_note": ("全样本评估，未覆盖行 play 特征为 NaN 交给 XGBoost；"
                                "不取交集以避免热门视频选择偏差（交集内正样本率 58%）"),
              "play_features": PLAY_FEATURES, "ratio_features": RATIO_FEATURES,
              "scenarios": {}}

    cov_te = covered.reindex(te.index).fillna(False).values.astype(bool)

    SCEN = [
        dict(key="abs", label="label_abs", name="绝对热度（次日进前25%）", naive="like_user_num"),
        dict(key="break", label="label_break", name="相对跃升（≥自身7日均值2倍）", naive="roll7_like"),
    ]

    for sc in SCEN:
        ytr, yte = tr[sc["label"]].values, te[sc["label"]].values
        naive = te[sc["naive"]].fillna(-1).values.astype(float)
        auc_naive = float(roc_auc_score(yte, naive))
        base20 = F.eval_per_day(naive, yte, days_te, 0.20)["recall"]
        print(f"\n[4] 口径：{sc['name']}  正样本率 {yte.mean()*100:.2f}%")
        print(f"    基线（按 {sc['naive']} 排序）"
              f"AUC {auc_naive:.4f}  top20% 召回 {base20:.2f}%")
        print(f"    {'组':<4}{'维度':>4}{'AUC':>9}{'R@20':>9}{'P@20':>9}{'vs基线':>9}{'ΔAUC':>9}")

        rows = {}
        auc_g0 = None
        for gid, name, feats in GROUPS:
            m = F.make_xgb(F.SEED)
            m.fit(tr[feats], ytr)
            p = m.predict_proba(te[feats])[:, 1]
            auc = float(roc_auc_score(yte, p))
            e20 = F.eval_per_day(p, yte, days_te, 0.20)
            # 只在 play_log 覆盖到的子集上单看 AUC（该子集正样本率约 58%，
            # top-k 会被机械封顶，不可用，故此处只报 AUC）
            auc_cov = float(roc_auc_score(yte[cov_te], p[cov_te]))
            if auc_g0 is None:
                auc_g0 = auc
            rows[gid] = dict(
                name=name, n_feat=len(feats), auc=round(auc, 4),
                auc_on_covered_subset=round(auc_cov, 4),
                top20_recall=round(e20["recall"], 2), top20_precision=round(e20["precision"], 2),
                lift_vs_naive=round(e20["recall"] - base20, 2),
                delta_auc_vs_G0=round(auc - auc_g0, 4),
            )
            print(f"    {gid:<4}{len(feats):>4}{auc:>9.4f}{e20['recall']:>8.2f}%"
                  f"{e20['precision']:>8.2f}%{e20['recall']-base20:>+8.2f}"
                  f"{auc-auc_g0:>+8.4f}")

        # 以 G0 的 top20 召回为基准，补上每组相对 G0 的增益
        g0_recall = rows["G0"]["top20_recall"]
        for gid in rows:
            rows[gid]["delta_recall_vs_G0"] = round(rows[gid]["top20_recall"] - g0_recall, 2)
        print(f"    → 相对 G0：播放分布 {rows['G1']['delta_recall_vs_G0']:+.2f}pp，"
              f"显式比率 {rows['G2']['delta_recall_vs_G0']:+.2f}pp，"
              f"全部 {rows['G3']['delta_recall_vs_G0']:+.2f}pp（top20% 召回）")
        print("    → 覆盖子集 AUC："
              + "  ".join(f"{gid} {rows[gid]['auc_on_covered_subset']:.4f}" for gid in rows))

        result["scenarios"][sc["key"]] = dict(
            name=sc["name"], pos_rate=round(float(yte.mean()) * 100, 2),
            naive_col=sc["naive"], naive_auc=round(auc_naive, 4),
            naive_top20_recall=round(base20, 2), groups=rows)

    with open(os.path.join(OUT, "exp_playlog.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\n已写 outputs/exp_playlog.json")

    # 特征重要性：看模型到底用没用这些新特征
    print("\n[5] G3 在跃升口径下的前 12 特征(gain):")
    feats = F.FEATURES + PLAY_FEATURES + RATIO_FEATURES
    m = F.make_xgb(F.SEED)
    m.fit(tr[feats], tr["label_break"].values)
    imp = sorted(zip(feats, m.feature_importances_), key=lambda t: -t[1])[:12]
    for k, v in imp:
        tag = "★play_log" if k in PLAY_FEATURES else ("★比率" if k in RATIO_FEATURES else "")
        print(f"    {k:<18}{v:>8.4f}  {tag}")


if __name__ == "__main__":
    main()
