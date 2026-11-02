# -*- coding: utf-8 -*-
"""特征工程 + 机器学习（时间切分）+ 图表生成
目标：用视频在 t 日及历史的表现，预测其 t+1 日点赞量是否进入当日活跃视频前 25%（“高潜内容”）
输出: outputs/model_metrics.json + charts/*.png
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymysql
from scipy.stats import norm

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = dict(host=os.environ.get("MYSQL_HOST", "127.0.0.1"),
          port=int(os.environ.get("MYSQL_PORT", "3307")),
          user=os.environ.get("MYSQL_USER", "root"),
          password=os.environ.get("MYSQL_PASSWORD", ""),
          database=os.environ.get("MYSQL_DB", "kuairec"))
SPLIT_TAIL_DAYS = 14          # 末尾 N 天作测试集（时间切分）
TOP_Q = 0.75                  # 高潜=次日点赞量≥当日第75分位
OUT = os.path.join(PROJ, "outputs")
CH = os.path.join(PROJ, "charts")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CH, exist_ok=True)


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


def build_dataset(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["d"].astype(str), format="%Y%m%d")
    df["dur_log"] = np.log1p(df["video_duration_ms"].fillna(0) / 1000.0)
    df["is_ad"] = (df["video_type"] == "AD").astype(int)
    df["up_type_idx"] = df["upload_type"].fillna("UNKNOWN").astype("category").cat.codes
    df["tag_idx"] = df["video_tag_name"].fillna("_NA_").astype("category").cat.codes
    df["dow"] = df["date"].dt.weekday
    df["like_rate"] = (df["like_user_num"] / df["play_user_num"].replace(0, np.nan)).fillna(0)
    df["complete_rate"] = (df["complete_play_cnt"] / df["play_cnt"].replace(0, np.nan)).fillna(0)

    g = df.sort_values("d").groupby("video_id")
    # t-1 / t-2 / t-3 滞后与 7 日历史均值（不含当日）
    for lag in (1, 2, 3):
        df[f"lag{lag}_like"] = g["like_user_num"].shift(lag)
        df[f"lag{lag}_playuser"] = g["play_user_num"].shift(lag)
    df["lag1_likerate"] = g["like_rate"].shift(1)
    df["roll7_like"] = g["like_user_num"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    df["roll7_playuser"] = g["play_user_num"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    df["roll7_complete"] = g["complete_play_cnt"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    df["n_prior_days"] = g.cumcount()
    df["gap_days"] = g["d"].diff().fillna(99)

    # 目标：次日(下一次活跃日)点赞量与日期
    df["next_d"] = g["d"].shift(-1)
    df["next_like"] = g["like_user_num"].shift(-1)
    df = df[df["next_d"].notna()].copy()          # 只有存在次日观测的行
    df["next_d"] = df["next_d"].astype(int)
    df["next_date"] = pd.to_datetime(df["next_d"].astype(str), format="%Y%m%d")

    # 标签：次日点赞量是否达到“当日活跃视频”前 25%
    day_q = df.groupby("next_d")["next_like"].transform(lambda s: s.quantile(TOP_Q))
    df["label"] = (df["next_like"] >= day_q).astype(int)
    return df


FEATURES = [
    "dur_log", "is_ad", "up_type_idx", "tag_idx", "dow",
    "play_user_num", "like_user_num", "complete_play_cnt", "like_rate", "complete_rate",
    "lag1_like", "lag2_like", "lag3_like", "lag1_playuser", "lag2_playuser", "lag3_playuser",
    "lag1_likerate", "roll7_like", "roll7_playuser", "roll7_complete", "n_prior_days", "gap_days",
]


def main():
    print("[1] 读取 video_daily ...")
    df = fetch_video_daily()
    print(f"    活跃视频日 {len(df):,} 行, {df['video_id'].nunique():,} 视频, {df['d'].nunique()} 天")

    print("[2] 构建样本/特征 ...")
    ds = build_dataset(df)
    ds = ds.dropna(subset=FEATURES)
    print(f"    可建模样本 {len(ds):,} 行；正样本(次日高潜)率 {ds['label'].mean()*100:.1f}%")
    day_min, day_max = ds["d"].min(), ds["d"].max()
    split_d = int((pd.to_datetime(str(day_max), format="%Y%m%d") - pd.Timedelta(days=SPLIT_TAIL_DAYS)).strftime("%Y%m%d"))
    tr = ds[ds["d"] <= split_d]
    te = ds[ds["d"] > split_d]
    print(f"    train: d<= {split_d} -> {len(tr):,} 行 | test: 后{SPLIT_TAIL_DAYS}天 -> {len(te):,} 行")

    Xtr, ytr = tr[FEATURES], tr["label"].values
    Xte, yte = te[FEATURES], te["label"].values

    print("[3] 训练 LR 与 XGBoost（时间切分）...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb

    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced"))
    lr.fit(Xtr, ytr)
    proba_lr = lr.predict_proba(Xte)[:, 1]

    xm = xgb.XGBClassifier(n_estimators=500, learning_rate=0.06, max_depth=6,
                           subsample=0.9, colsample_bytree=0.9, eval_metric="auc",
                           tree_method="hist", n_jobs=-1, random_state=42)
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
    top_recall = {r["k_pct"]: round(r["recall"] * 100, 2) for r in rows}
    print("    top-k 圈选召回率:", top_recall)

    fpr, tpr, _ = roc_curve(yte, proba_xgb)
    # 基线：次日高潜若随机命中，top20% 召回=20%
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
        importance=[],
    )
    # 特征重要性（gain）
    imp = sorted(zip(FEATURES, xm.feature_importances_), key=lambda t: -t[1])
    metrics["importance"] = [dict(feat=f, gain=round(float(g), 4)) for f, g in imp[:12]]
    print("    top 特征:", [i["feat"] for i in metrics["importance"][:6]])

    with open(os.path.join(OUT, "model_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # ---------------- 图表 ----------------
    print("[4] 画图 ...")
    # ROC + PR
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    fpr_lr, tpr_lr, _ = roc_curve(yte, proba_lr)
    axes[0].plot(fpr_lr, tpr_lr, label=f"逻辑回归 (AUC={auc_lr:.3f})", lw=1.8)
    axes[0].plot(fpr, tpr, label=f"XGBoost (AUC={auc_xgb:.3f})", lw=1.8)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=.5)
    axes[0].set(title="次日高潜内容识别 ROC（时间切分测试集）", xlabel="FPR", ylabel="TPR")
    axes[0].legend()
    pr_lr_p, pr_lr_r, _ = precision_recall_curve(yte, proba_lr)
    axes[1].plot(pr_lr_r, pr_lr_p, label=f"逻辑回归 (AP={ap_lr:.3f})", lw=1.8)
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
