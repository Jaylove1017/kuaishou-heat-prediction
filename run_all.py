# -*- coding: utf-8 -*-
"""一键复跑：加载数据 → SQL 分析 → 建模/图表

用法:
    python run_all.py

数据库连接: 默认 127.0.0.1:3307 root/空密码/kuairec，
可用环境变量 MYSQL_HOST / MYSQL_PORT / MYSQL_USER / MYSQL_PASSWORD / MYSQL_DB 覆盖。

前置条件：先把 KuaiRec 的两个原始文件放入 data/raw/，详见 README「数据」一节。
"""
import os
import subprocess
import sys

PROJ = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def step(name, script):
    print(f"\n===== {name} =====")
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run([PY, "-u", os.path.join(PROJ, "src", script)], env=env)
    if r.returncode != 0:
        raise SystemExit(f"[{script}] 失败，退出码 {r.returncode}")


def main():
    step("1. 清洗并入库 (MySQL)", "load_data.py")
    step("2. SQL 业务指标", "run_analysis.py")
    step("3. 特征工程 + 机器学习 + 模型图", "model_analysis.py")
    step("4. 业务洞察图表", "charts_overview.py")
    print("\n完成。产物见 outputs/ 与 charts/。")
    print("延伸实验请单独运行 experiments/ 下的脚本，见 README「实验设计」。")


if __name__ == "__main__":
    main()
