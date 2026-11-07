# -*- coding: utf-8 -*-
"""一键复跑：加载数据 → SQL 分析 → 建模/图表

用法:
    python run_all.py                     # 完整流水线
    python run_all.py --skip-load         # 数据已入库，跳过清洗与灌库（改 SQL/建模时用）
    python run_all.py --with-experiments  # 额外跑 3 个延伸实验（较慢）

数据库连接: 默认 127.0.0.1:3307 root/空密码/kuairec，
可用环境变量 MYSQL_HOST / MYSQL_PORT / MYSQL_USER / MYSQL_PASSWORD / MYSQL_DB 覆盖。

前置条件：先把 KuaiRec 的两个原始文件放入 data/raw/，详见 README「快速开始」。
"""
import argparse
import os
import subprocess
import sys

PROJ = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
RAW_FILES = ("big_matrix.csv.gz", "item_daily_features.csv.gz")


def precheck():
    """缺数据就给出能直接照做的提示，而不是等到 gzip.open 抛 FileNotFoundError"""
    raw = os.path.join(PROJ, "data", "raw")
    missing = [f for f in RAW_FILES if not os.path.exists(os.path.join(raw, f))]
    if missing:
        msg = [
            "缺少原始数据：" + ", ".join(missing),
            f"请下载到 {raw}/，获取方式见 data/README.md。",
            "",
            "一条命令搞定（约 350 MB）：",
            "  pip install huggingface_hub",
            "  huggingface-cli download hiiamkik/kuai-rec-data --repo-type dataset \\",
            '    --include "big_matrix.csv.gz" "item_daily_features.csv.gz" --local-dir data/raw',
        ]
        raise SystemExit("\n".join(msg))


def step(name, script):
    print(f"\n===== {name} =====")
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    path = os.path.join(PROJ, script)
    if not os.path.exists(path):
        raise SystemExit(f"[{script}] 不存在")
    r = subprocess.run([PY, "-u", path], env=env)
    if r.returncode != 0:
        raise SystemExit(f"[{script}] 失败，退出码 {r.returncode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-load", action="store_true", help="跳过清洗与灌库")
    ap.add_argument("--with-experiments", action="store_true",
                    help="额外运行 eval_upgrade / ablation / significance")
    args = ap.parse_args()

    if not args.skip_load:
        precheck()
        step("1. 清洗并入库 (MySQL)", os.path.join("src", "load_data.py"))
    else:
        print("[skip] 清洗与灌库已跳过")

    step("2. SQL 业务指标", os.path.join("src", "run_analysis.py"))
    step("3. 特征工程 + 机器学习 + 模型图", os.path.join("src", "model_analysis.py"))
    step("4. 业务洞察图表", os.path.join("src", "charts_overview.py"))

    if args.with_experiments:
        step("5. 评估口径升级（per-day）", os.path.join("experiments", "eval_upgrade.py"))
        step("6. 特征消融（8 组）", os.path.join("experiments", "ablation.py"))
        step("7. 显著性检验（bootstrap + 多种子）", os.path.join("experiments", "significance.py"))
        step("8. 口径审计（严格次日 / 冷启动 / 切分泄漏）",
             os.path.join("experiments", "strict_eval.py"))
        step("9. 播放日志特征（负结果）", os.path.join("experiments", "playlog_features.py"))

    print("\n完成。产物见 outputs/ 与 charts/。")
    print("其余实验请单独运行：")
    print("  python experiments/breakout_experiment.py  # 跃升标签 + 方向检验")
    print("  python experiments/baseline_vs_model.py    # 旧口径分段（勿与主表混引）")
    print("  python experiments/plot_ablation.py        # 只读 json 重出图")


if __name__ == "__main__":
    main()
