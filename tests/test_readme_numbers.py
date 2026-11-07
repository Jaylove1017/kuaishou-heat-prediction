# -*- coding: utf-8 -*-
"""README 与 outputs/*.json 的一致性检查。

为什么要有这个文件
------------------
本项目出过一次真实事故：README 总表写「top20 召回 77.01% / 77.22%」，
而它标注的来源 `baseline_vs_model.py` 算出来是 76.17% / 76.39% ——
前者属于 per-day 口径、后者属于全局混排口径，两个数被拼进了同一张表，
差值刚好等于「模型 vs 基线」那一栏的量级。任何认真核对 json 的人都会当场发现。

这个测试把「README 的数字必须能在 json 里找到」变成 CI 的一部分：
改脚本重跑后如果数字变了而 README 没跟着改，构建会红。

注意：这是**故意**会让 CI 失败的设计。重跑得到新数字时，正确做法是同步更新
README 与 outputs/README.md，而不是放宽这个测试。
"""
import csv
import json
import os

import pytest

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(PROJ, "README.md")
OUT = os.path.join(PROJ, "outputs")


def _readme():
    if not os.path.exists(README):
        pytest.skip("没有 README.md")
    with open(README, encoding="utf-8") as f:
        return f.read()


def _json(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        pytest.skip(f"没有 {name}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def test_headline_numbers_match_eval_upgrade():
    """总表：AUC 与 per-day top20 召回必须是 exp_eval_upgrade.json 里的数"""
    j = _json("exp_eval_upgrade.json")
    a = j["scenarios"]["abs"]
    readme = _readme()

    checks = {
        "模型 AUC": f"{a['auc']['xgb']:.4f}",
        "基线 AUC": f"{a['auc']['naive']:.4f}",
        "模型 top20 召回": f"{a['perday_topk']['20']['model']['recall']:.2f}%",
        "基线 top20 召回": f"{a['perday_topk']['20']['naive']['recall']:.2f}%",
        "跃升口径模型 AUC": f"{j['scenarios']['break']['auc']['xgb']:.4f}",
        "跃升口径基线 AUC": f"{j['scenarios']['break']['auc']['naive']:.4f}",
    }
    missing = {k: v for k, v in checks.items() if v not in readme}
    assert not missing, (
        "README 里找不到下列数字（json 变了但 README 没同步？）：\n"
        + "\n".join(f"  {k}: {v}" for k, v in missing.items())
    )


def test_topk_table_matches_json():
    """§3.1 的全 k 表逐行核对（模型/基线召回）"""
    j = _json("exp_eval_upgrade.json")["scenarios"]["abs"]["perday_topk"]
    readme = _readme()
    bad = []
    for k in ("5", "10", "15", "20", "25", "30"):
        m = f"{j[k]['model']['recall']:.2f}%"
        n = f"{j[k]['naive']['recall']:.2f}%"
        if m not in readme or n not in readme:
            bad.append(f"top{k}%: 模型 {m} / 基线 {n}")
    assert not bad, "README 的 top-k 表与 json 不一致：\n" + "\n".join(bad)


def test_ablation_numbers_match_json():
    """§4.1 与 §4.4 引用的消融 AUC"""
    j = _json("exp_ablation.json")["scenarios"]
    readme = _readme()
    bad = []
    for scen, gids in (("abs", ("G1", "G2", "G3", "G5", "G7", "G8")),
                       ("break", ("G1", "G2", "G3", "G5", "G7", "G8"))):
        for g in gids:
            v = f"{j[scen]['groups'][g]['auc_xgb']:.4f}"
            if v not in readme:
                bad.append(f"{scen}/{g} = {v}")
    assert not bad, "README 的消融数字与 json 不一致：\n" + "\n".join(bad)


def test_robustness_table_matches_csv():
    """§5 的标签稳健性表来自 label_robustness.csv"""
    p = os.path.join(OUT, "label_robustness.csv")
    if not os.path.exists(p):
        pytest.skip("没有 label_robustness.csv")
    with open(p, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    readme = _readme()
    bad = []
    for r in rows:
        for col in ("AUC", "基线AUC"):
            v = f"{float(r[col]):.4f}"
            if v not in readme:
                bad.append(f"{r['口径']} {col} = {v}")
    assert not bad, "README 的稳健性表与 csv 不一致：\n" + "\n".join(bad)


def test_no_deprecated_global_top20_as_headline():
    """旧口径的 76.17% 不能出现在总表区域（它只允许出现在 §3.3 的口径对照里）"""
    readme = _readme()
    if "76.17%" not in readme:
        return
    # 允许出现，但必须同时出现「全局」字样说明它是什么口径
    idx = readme.index("76.17%")
    window = readme[max(0, idx - 300): idx + 300]
    assert "全局" in window, (
        "README 里出现了旧口径的 76.17% 但没有注明「全局混排」，"
        "容易被误读成 per-day 的主结论数字")
