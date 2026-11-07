# -*- coding: utf-8 -*-
"""schema.sql 与 load_data.py 的列定义必须严格对齐。

这条测试针对一个真实发生过的 bug：schema 里 video_daily 有 33 列，
而清洗脚本的 usecols 只有 32 列（漏了 click_like_cnt），
LOAD DATA 又不指定列名 —— MySQL 按位置灌入，从第 30 列起整体错一列，且不报错。
核心标签 like_user_num 恰好排在错位点之前，所以模型结论没被污染，
但双击/取消赞三个指标全错，任何用到它们的 SQL 都不可信。

现在两处都对齐了，这个测试保证以后改任何一侧都会被拦下来。
不需要 MySQL，纯文本解析。
"""
import os
import re

import pytest

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA = os.path.join(PROJ, "sql", "schema.sql")

try:
    from src.load_data import PLAY_COLS, VD_COLS
except ImportError:  # pragma: no cover
    pytest.skip("无法 import src.load_data", allow_module_level=True)

NOT_COLUMNS = ("KEY", "PRIMARY", "UNIQUE", "INDEX", "CONSTRAINT", "FOREIGN")


def _table_columns(table):
    with open(SCHEMA, encoding="utf-8") as f:
        sql = f.read()
    m = re.search(r"CREATE TABLE " + table + r" \((.*?)\) ENGINE", sql, re.S)
    assert m, f"schema.sql 里找不到 CREATE TABLE {table}"
    cols = []
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("--"):
            continue
        if line.upper().startswith(NOT_COLUMNS):
            continue
        cols.append(line.split()[0])
    return cols


def test_video_daily_columns_match():
    table = _table_columns("video_daily")
    loader = [t for _, t in VD_COLS]
    assert len(table) == len(loader), (
        f"video_daily 有 {len(table)} 列，清洗脚本期望 {len(loader)} 列。\n"
        f"表独有: {sorted(set(table) - set(loader))}\n"
        f"脚本独有: {sorted(set(loader) - set(table))}\n"
        "（这正是历史上导致 LOAD DATA 静默错位一列的原因）")
    assert table == loader, (
        "列名一致但顺序不同：LOAD DATA 虽然已显式给列名，顺序仍应与表定义保持可读的一致")


def test_play_log_columns_match():
    assert _table_columns("play_log") == PLAY_COLS


def test_no_duplicate_columns():
    for t in ("play_log", "video_daily"):
        cols = _table_columns(t)
        assert len(cols) == len(set(cols)), f"{t} 有重复列名"
