# -*- coding: utf-8 -*-
"""从 outputs/base_metrics.json 画业务洞察图（漏斗/时长/时段/分日/广告对比/用户分层）"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
# 跨平台中文字体回退（原实现只写了 Windows 字体，Linux/macOS 上出图全是方块）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC",
                                   "Heiti SC", "Noto Sans CJK SC",
                                   "Source Han Sans SC", "WenQuanYi Micro Hei",
                                   "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CH = os.path.join(PROJ, "charts")
os.makedirs(CH, exist_ok=True)


def wan(v):
    return v / 1e4


def yi(v):
    return v / 1e8


def main():
    with open(os.path.join(PROJ, "outputs", "base_metrics.json"), encoding="utf-8") as f:
        M = json.load(f)

    # ---- 1 内容侧漏斗（用户口径）----
    fun = M["内容侧漏斗"]
    stages = ["曝光用户", "播放用户", "有效播放用户", "完播用户", "点赞用户"]
    vals = [fun["show_user"], fun["play_user"], fun["valid_user"], fun["complete_user"], fun["like_user"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colors = ["#BDC3C7", "#85C1E9", "#5DADE2", "#2E86C1", "#E74C3C"]
    bars = ax.bar(stages, vals, color=colors, width=0.62)
    ax.set_yscale("log")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v/1e8:.2f}亿",
                ha="center", va="bottom", fontsize=10)
    ax.set_title("内容侧互动漏斗（全期用户曝光/播放/完播/点赞，对数轴）")
    ax.set_ylabel("次数（对数轴）")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(CH, "funnel.png"), dpi=150)
    plt.close(fig)

    # ---- 2 时长分箱对完播/点赞影响 ----
    dur = M["时长对完播影响"]
    order = ["<30s", "30-60s", "1-3min", "3-10min", ">10min"]
    rows = {r["dur_bucket"]: r for r in dur}
    labels = [k for k in order if k in rows]
    comp = [rows[k]["complete_pct"] for k in labels]
    like = [rows[k]["like_per_playuser_pct"] for k in labels]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(labels))
    ax.bar(x - 0.19, comp, 0.38, label="完播率 %", color="#2E86C1")
    ax2 = ax.twinx()
    ax2.bar(x + 0.19, like, 0.38, label="点赞/播放用户 %", color="#E67E22", alpha=0.85)
    ax.set_xticks(x, labels)
    ax.set_xlabel("视频时长分箱")
    ax.set_ylabel("完播率 %")
    ax2.set_ylabel("点赞/播放用户 %")
    for xi, v in zip(x - 0.19, comp):
        ax.text(xi, v + 0.4, f"{v:.1f}", ha="center", fontsize=9)
    for xi, v in zip(x + 0.19, like):
        ax2.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_title("不同时长视频的完播率与点赞表现")
    ax.grid(axis="y", alpha=0.3)
    fig.legend(loc="upper right", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(CH, "duration_completion.png"), dpi=150)
    plt.close(fig)

    # ---- 3 时段活跃（播放侧）----
    hr = M["时段活跃"]
    hrs = [r["hour"] for r in hr]
    plays = [r["plays"] for r in hr]
    comp_h = [r["complete_plays"] for r in hr]
    peak = hrs[int(np.argmax(plays))]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(hrs, [wan(p) for p in plays], color="#5DADE2", width=0.8, label="播放量(万)")
    ax.plot(hrs, [c / p * 100 for c, p in zip(comp_h, plays)], color="#E74C3C", marker="o",
            lw=1.8, label="完播率 %")
    ax.axvline(peak, color="gray", ls="--", lw=1, alpha=0.6)
    ax.annotate(f"峰值 {peak}:00", (peak, max(plays) / 1e4 * 0.99),
                xytext=(peak - 6, max(plays) / 1e4 * 0.9), fontsize=9, color="#C0392B")
    ax.set(xlabel="播放时段（北京时间，小时）", ylabel="播放量（万）",
           title="播放时段分布与完播率")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(CH, "hourly.png"), dpi=150)
    plt.close(fig)

    # ---- 4 分日播放/点赞趋势 ----
    day = M["分日播放与活跃"]
    dd = pd.DataFrame(day)
    dd["dt"] = pd.to_datetime(dd["d"].astype(str), format="%Y%m%d")
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    ax.plot(dd["dt"], wan(dd["play_cnt"]), color="#2E86C1", lw=1.6, label="日播放量(万)")
    ax2 = ax.twinx()
    ax2.plot(dd["dt"], dd["like_cnt"], color="#E67E22", lw=1.2, label="日点赞数")
    ax.set_title("逐日播放与点赞趋势（2020-07-05 ~ 09-05）")
    ax.set_ylabel("播放量（万）"); ax2.set_ylabel("点赞数")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(CH, "daily_trend.png"), dpi=150)
    plt.close(fig)

    # ---- 5 广告 vs 普通内容 ----
    adv = M["广告 vs 普通内容互动"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    names = [r["video_type"] for r in adv]
    comp_a = [r["complete_pct"] for r in adv]
    avg_p = [r["avg_play"] for r in adv]
    ax.bar(names, comp_a, color=["#7FB3D5", "#F1948A"], width=0.5, label="完播率 %")
    for xi, v in zip(range(len(names)), comp_a):
        ax.text(xi, v + 0.3, f"{v:.2f}%", ha="center")
    ax.set_ylabel("完播率 %")
    ax.set_title("广告内容 vs 普通内容完播率")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(CH, "ad_vs_normal.png"), dpi=150)
    plt.close(fig)

    # ---- 6 用户完播分层 ----
    seg = M["用户完播率分布"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    names = [r["seg"] for r in seg]
    u = [r["users"] for r in seg]
    bars = ax.bar(names, u, color=["#16A085", "#2E86C1", "#BDC3C7"], width=0.55)
    for b, v in zip(bars, u):
        ax.text(b.get_x() + b.get_width() / 2, v + 40, f"{v:,}", ha="center")
    ax.set_title("用户完播分层（按个人完播率 ratio≥1 占比）")
    ax.set_ylabel("用户数")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(CH, "user_segments.png"), dpi=150)
    plt.close(fig)

    print("charts 已保存到", CH)


if __name__ == "__main__":
    main()
