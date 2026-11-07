# -*- coding: utf-8 -*-
"""特征工程 + 机器学习（时间切分）+ 图表生成

目标：用视频在 t 日及历史的表现，预测其 t+1 日点赞量是否进入当日可预测样本前 25%（“高潜内容”）
输出: outputs/model_metrics.json + charts/*.png

特征与标签的实现全部来自 src/features.py —— 全项目唯一一份，本文件只负责取数、训练、出图。
历史上这里的 build_dataset 与 experiments/ 下四个脚本各存了一份副本，已经漂移过
（两条数据通路 + n_jobs 不一致），所以收敛到 features.py。

命令行开关（默认全部关闭，保持与已提交的 outputs/ 逐位一致）：
    --source csv       改从 data/clean/video_daily.csv 读（默认走 MySQL）
    --strict-gap       只保留 next_d == d+1 的严格次日样本
    --strict-split     按目标日 next_d 切分，而非特征日 d
    --keep-na          不 dropna，冷启动样本交给 XGBoost 原生缺失值处理
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymysql

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import features as F

sys.stdout.reconfigure(encoding="utf-8")
F.setup_style()

PROJ = F.PROJ
DB = dict(host=os.environ.get("MYSQL_HOST", "127.0.0.1"),
          port=int(os.environ.get("MYSQL_PORT", "3307")),
          user=os.environ.get("MYSQL_USER", "root"),
          password=os.environ.get("MYSQL_PASSWORD", ""),
          database=os.environ.get("MYSQL_DB", "kuairec"))
OUT = os.path.join(PROJ, "outputs")
CH = os.path.join(PROJ, "charts")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CH, exist_ok=True)

FEATURES = F.FEATURES


def fetch_video_daily():
    conn = pymysql.connect(host=DB["host"], port=DB["port"], user=DB["user"],
                           password=DB["password"], database=DB["database"], charset="utf8mb4")
    sql = """
    SELECT video_id, d, author_id, video_type, upload_type, upload_dt, video_duration_ms,
           video_tag_name, play_cnt, play_user_num, complete_play_cnt, like_cnt, like_user_num
    FROM video_daily
    WHERE play_cnt > 0 AND d >= 20200705
    ORDER BY video_id, d
    """
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["mysql", "csv"], default="mysql")
    ap.add_argument("--strict-gap", action="store_true",
                    help="只保留 next_d == d+1 的严格次日样本")
    ap.add_argument("--strict-split", action="store_true",
                    help="按目标日 next_d 切分（默认按特征日 d）")
    ap.add_argument("--keep-na", action="store_true",
                    help="不 dropna，保留冷启动样本")
    args = ap.parse_args()

    print("[1] 读取 video_daily ...")
    if args.source == "csv":
        df = F.load_daily_csv()
    else:
        df = fetch_video_daily()
    df = F.normalize(df)
    print(f"    活跃视频日 {len(df):,} 行, {df['video_id'].nunique():,} 视频, {df['d'].nunique()} 天")

    print("[2] 构建样本/特征 ...")
    gap_max = 1 if args.strict_gap else None
    ds = F.build_dataset(df, dropna=not args.keep_na, gap_max=gap_max)
    print(f"    可建模样本 {len(ds):,} 行；正样本(次日高潜)率 {ds['label_abs'].mean()*100:.1f}%")
    if args.keep_na:
        print(f"    其中冷启动样本(n_prior_days<3) {int((ds['n_prior_days'] < 3).sum()):,} 行")

    split_on = "next_d" if args.strict_split else "d"
    tr, te, split_d = F.split(ds, on=split_on)
    day_max = ds["d"].max()
    print(f"    train({split_on}<={split_d}): {len(tr):,} 行 | test: 后14天 -> {len(te):,} 行")

    Xtr = tr[FEATURES]
    ytr = tr["label_abs"].values
    Xte = te[FEATURES]
    yte = te["label_abs"].values

    print("[3] 训练 LR 与 XGBoost（时间切分）...")
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 roc_curve, precision_recall_curve)

    lr = F.make_lr()
    lr.fit(Xtr, ytr)
    proba_lr = lr.predict_proba(Xte)[:, 1]

    xm = F.make_xgb()
    xm.fit(Xtr, ytr)
    proba_xgb = xm.predict_proba(Xte)[:, 1]

    auc_lr = roc_auc_score(yte, proba_lr)
    auc_xgb = roc_auc_score(yte, proba_xgb)
    ap_lr = average_precision_score(yte, proba_lr)
    ap_xgb = average_precision_score(yte, proba_xgb)
    print(f"    LR  AUC={auc_lr:.4f} AP={ap_lr:.4f}")
    print(f"    XGB AUC={auc_xgb:.4f} AP={ap_xgb:.4f}")

    # 按概率排序圈选 top k% 视频日的召回/命中率
    order = np.argsort(-proba_xgb)
    n_test = len(yte)
    rows = []
    for k in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        topk = order[: int(n_test * k)]
        hits = yte[topk].sum()
        rows.append(dict(k_pct=k * 100,
                         recall=hits / max(yte.sum(), 1),
                         precision=hits / max(len(topk), 1)))
    top_recall = {r["k_pct"]: round(float(r["recall"]) * 100, 2) for r in rows}
    print("    top-k 圈选召回率:", top_recall)

    fpr, tpr, _ = roc_curve(yte, proba_xgb)
    # 随机基线：均匀抽 top20% 的期望召回就是 20%
    base_top20 = 20.0

    metrics = dict(
        model_rows=int(len(ds)),
        train_rows=int(len(tr)), test_rows=int(len(te)),
        test_days=int(len(te["d"].unique())),
        pos_rate_train=round(float(ytr.mean()) * 100, 2),
        pos_rate_test=round(float(yte.mean()) * 100, 2),
        auc_lr=round(float(auc_lr), 4), auc_xgb=round(float(auc_xgb), 4),
        ap_lr=round(float(ap_lr), 4), ap_xgb=round(float(ap_xgb), 4),
        topk_recall=top_recall,
        baseline_random_top20=base_top20,
        split_day=int(split_d), max_day=int(day_max),
        features=len(FEATURES),
        # 口径元信息：让复跑的人一眼看出这份指标是在什么口径下算的
        split_on=split_on, gap_max=gap_max, dropna=not args.keep_na,
        seed=F.SEED, n_jobs=F.N_JOBS,
        importance=[],
    )
    # 特征重要性（gain —— XGBClassifier 默认是 weight，这里已显式指定 importance_type="gain"）
    imp = sorted(zip(FEATURES, xm.feature_importances_), key=lambda t: -t[1])
    metrics["importance"] = [dict(feat=f, gain=round(float(g), 4)) for f, g in imp[:12]]
    print("    top 特征:", [i["feat"] for i in metrics["importance"][:6]])

    with open(os.path.join(OUT, "model_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # ---------------- 图表 ----------------
    print("[4] 画图 ...")
    # ROC + PR（两条线都要画：只画 LR 会让人误以为那条是主模型）
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    fpr_lr, tpr_lr, _ = roc_curve(yte, proba_lr)
    axes[0].plot(fpr_lr, tpr_lr, label=f"逻辑回归 (AUC={auc_lr:.3f})", lw=1.8)
    axes[0].plot(fpr, tpr, label=f"XGBoost (AUC={auc_xgb:.3f})", lw=1.8)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=.5)
    axes[0].set(title="次日高潜内容识别 ROC（时间切分测试集）", xlabel="FPR", ylabel="TPR")
    axes[0].legend()

    pr_lr_p, pr_lr_r, _ = precision_recall_curve(yte, proba_lr)
    pr_xgb_p, pr_xgb_r, _ = precision_recall_curve(yte, proba_xgb)
    axes[1].plot(pr_lr_r, pr_lr_p, label=f"逻辑回归 (AP={ap_lr:.3f})", lw=1.8)
    axes[1].plot(pr_xgb_r, pr_xgb_p, label=f"XGBoost (AP={ap_xgb:.3f})", lw=1.8)
    axes[1].axhline(yte.mean(), color="gray", ls="--", lw=1, alpha=.7, label="基线(随机)")
    axes[1].set(title="PR 曲线", xlabel="Recall", ylabel="Precision")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(CH, "roc_pr.png"), dpi=150)
    plt.close(fig)

    # top-k 圈选召回
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ks = [r["k_pct"] for r in rows]
    rs = [r["recall"] * 100 for r in rows]
    ax.plot(ks, rs, "o-", color="#2980B9", lw=2)
    ax.plot([0, 30], [0, 30], "k--", alpha=.45, label="随机圈选基线")
    ax.scatter([20], [top_recall[20.0]], s=90, color="#E74C3C", zorder=5)
    ax.annotate(f"top20% 圈选\n召回 {top_recall[20.0]:.0f}%", (20, top_recall[20.0]),
                textcoords="offset points", xytext=(8, -18), color="#E74C3C")
    ax.set(title="按预测分排序圈选“高潜内容”的召回曲线", xlabel="圈选比例（Top-k%）", ylabel="高潜命中率（召回）%")
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(CH, "topk_recall.png"), dpi=150)
    plt.close(fig)

    # 特征重要性 top10
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    topi = metrics["importance"][:10][::-1]
    ax.barh([i["feat"] for i in topi], [i["gain"] for i in topi], color="#16A085")
    ax.set(title="XGBoost 特征重要性（gain，Top10）", xlabel="gain")
    fig.tight_layout()
    fig.savefig(os.path.join(CH, "feature_importance.png"), dpi=150)
    plt.close(fig)
    print("saved outputs/model_metrics.json & charts")


if __name__ == "__main__":
    main()
