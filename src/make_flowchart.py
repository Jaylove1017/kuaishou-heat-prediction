# -*- coding: utf-8 -*-
"""按文件名绘制快手项目数据管道流程图 → 项目根目录 PNG
用法: python src/make_flowchart.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PNG = os.path.join(PROJ, "流程图_快手项目数据管道.png")

C_BG = "#F7F9FB"
COLOR = {
    "h": "#1A3352",  # 主脚本/SQL
    "f": "#117A4B",  # 数据/源码文件名
    "n": "#6B7B8C",  # 说明
    "o": "#B9770E",  # 中间产出
    "m": "#C0392B",  # 最终交付
}

CARDS = [
    dict(name="run_all.py", lines=[
        ("h", "run_all.py   ——   一键复跑总入口（项目根目录）"),
        ("f", "python run_all.py             （可选 python run_all.py --docs 追加渲染简历）"),
        ("n", "顺序执行 ①→④ 后自动出图/汇总；数据库连接用环境变量 MYSQL_HOST/PORT/USER/PASSWORD 可指向你自己的 MySQL"),
    ]),
    dict(name="① 数据源  data/raw", lines=[
        ("f", "data/raw/big_matrix.csv.gz            播放日志 × 12,530,806（28 天；7,176 用户 × 10,728 视频）"),
        ("f", "data/raw/item_daily_features.csv.gz   内容×日统计 × 343,341（63 天）"),
        ("n", "来源：快手 KuaiRec 公开论文数据集（huggingface: hiiamkik/kuai-rec-data，本机经 hf-mirror 下载）"),
    ]),
    dict(name="② 清洗入库", lines=[
        ("h", "sql/schema.sql    ——   建库建表 kuairec（utf8mb4；play_log / video_daily + 索引）"),
        ("h", "src/load_data.py  ——   流式清洗 + LOAD DATA 灌库（剔除异常约 2.2%；完播倍率≥10 标记为异常流）"),
        ("o", "中间产物 →  data/clean/play_log.csv、data/clean/video_daily.csv"),
        ("o", "产出     →  MySQL kuairec 库：play_log 12,290,655 行  /  video_daily 343,341 行"),
    ]),
    dict(name="③ 业务指标 SQL", lines=[
        ("h", "src/run_analysis.py   ——   逐段执行 sql/analysis.sql，结果归档为 json"),
        ("h", "sql/analysis.sql       ——   13 组聚合：规模/完播分布/漏斗/口径对照/头部1%/广告vs普通/时长/分日/时段/用户分层/高潜候选量级"),
        ("o", "产出 →  outputs/base_metrics.json（完播 34.3%、广告 vs 普通 5.5% vs 54.0% 等真实数字）"),
    ]),
    dict(name="④ 机器学习建模", lines=[
        ("h", "src/model_analysis.py   ——   从 MySQL 取数；22 维特征（滞后 1-3 日 + 7 日滚动）；时间切分；LR vs XGBoost"),
        ("f", "标签口径：次日点赞 ≥ 当日活跃内容前 25%；训练 18.6 万行 / 后 14 天测试 9.3 万行 → AUC 0.946 / 0.982"),
        ("o", "产出 →  outputs/model_metrics.json + charts/ 下模型图 3 张（roc_pr、topk_recall、feature_importance）"),
    ]),
    dict(name="业务图表（可与 ④ 并行理解）", lines=[
        ("h", "src/charts_overview.py   ——   读 outputs/base_metrics.json 出业务图"),
        ("f", "charts/：funnel · duration_completion · hourly · daily_trend · ad_vs_normal · user_segments（6 张）"),
    ]),
    dict(name="⑤ 汇总 + 简历文案", lines=[
        ("h", "src/render_resume.py   ——   合并两份 metrics，自动套模板生成 3 条要点（改文案只改这里，数字不手填）"),
        ("o", "outputs/metrics.json + outputs/project1_text.txt（文案纯文本）"),
        ("m", "项目一_快手短视频内容互动分析.docx          （独立粘贴段）"),
        ("m", "简历_黄瑞杰_数据分析_新项目一.docx           （整份副本，仅替换“项目一”4 段）"),
    ]),
    dict(name="⑥ 转 PDF", lines=[
        ("h", "src/make_pdf.py   ——   docx → PDF（Word COM），顺带渲染 PNG 做排版检查"),
        ("m", "两个同名 .pdf 文件（A4 单页已验收：整简历恰好 1 页、无裁切）"),
    ]),
]


def main():
    X0, X1 = 2.2, 97.8
    W = X1 - X0
    Y0 = 158.0          # 顶部起点（大 y = 上）
    GAP = 6.6

    fig = plt.figure(figsize=(16, 22.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_facecolor(C_BG)

    y = Y0
    ax.text(50, y - 1.6, "快手短视频内容互动分析 · 数据管道流程图（按文件）",
            ha="center", va="top", fontsize=16, fontweight="bold", color="#1A3352")
    y -= 7.2

    prev_bottom = None
    for card in CARDS:
        n = len(card["lines"])
        h = 5.0 + n * 2.7
        top, bottom = y, y - h
        # 卡片
        ax.add_patch(FancyBboxPatch((X0, bottom), W, h,
                                    boxstyle="round,pad=0.35,rounding_size=1.1",
                                    linewidth=1.1, edgecolor="#AEB6BF", facecolor="white", zorder=2))
        # 顶条
        ax.add_patch(FancyBboxPatch((X0 + 0.5, bottom + h - 3.15), W - 1.0, 2.8,
                                    boxstyle="round,pad=0.15,rounding_size=0.8",
                                    linewidth=0, facecolor="#1A3352", zorder=3))
        ax.text(X0 + 1.5, bottom + h - 1.75, card["name"], va="center", ha="left",
                fontsize=11.5, fontweight="bold", color="white", zorder=4)
        # 正文
        yy = bottom + h - 5.4
        for kind, text in card["lines"]:
            ax.text(X0 + 2.0, yy, text, va="center", ha="left", fontsize=9.6,
                    color=COLOR[kind], zorder=4)
            yy -= 2.7
        # 卡片间箭头
        if prev_bottom is not None:
            ax.annotate("", xy=(50, top + GAP - 2.4), xytext=(50, prev_bottom - GAP + 2.4),
                        arrowprops=dict(arrowstyle="-|>", color="#566573", lw=2.4), zorder=5)
        prev_bottom = bottom
        y = bottom - GAP

    # 图例
    ly = y - 1.0
    ax.text(X0 + 0.5, ly, "图例：", fontsize=10.5, fontweight="bold", color="#34495E", va="center")
    lx = X0 + 9
    for kind, label in (("h", "主脚本 / SQL 文件"), ("f", "数据或源码文件名"), ("n", "说明 / 口径"),
                        ("o", "中间产出 json / csv"), ("m", "最终交付 docx / pdf")):
        ax.text(lx, ly, "■", fontsize=9, color=COLOR[kind], va="center")
        ax.text(lx + 1.7, ly, label, fontsize=9.8, color="#2C3E50", va="center")
        lx += 17.2

    ax.set_xlim(0, 100)
    ax.set_ylim(ly - 5, Y0 + 3)
    fig.savefig(OUT_PNG, dpi=200, facecolor=C_BG)
    plt.close(fig)
    print("saved:", OUT_PNG, f"{os.path.getsize(OUT_PNG)/1024:.0f}KB")


if __name__ == "__main__":
    main()
