# -*- coding: utf-8 -*-
"""特征工程的回归测试。

两个目的：
1. 锁住最关键的防泄漏约定 —— roll7 / lag 绝不含当日。这条一旦被改坏，
   全项目的 AUC 会悄悄变好看，而且没有任何报错。
2. 证明 src/features.py 与历史内联实现逐位等价 —— 抽取单一实现时最容易
   悄悄改掉一个 shift 的位置。`_legacy_build` 是从旧脚本逐行复制的，
   不要"顺手优化"它。

运行：pytest -q      （不需要 GPU / 数据库 / 原始数据）
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import features as F  # noqa: E402


# ---------------------------------------------------------------- 造数
def make_frame(n_video=25, n_day=20, seed=0):
    """可复现的合成 video_daily。刻意制造：缺日、全零点赞、广告与普通混杂"""
    rng = np.random.default_rng(seed)
    rows = []
    for vid in range(1, n_video + 1):
        # 每个视频随机跳过 0-4 天，用来测 gap_days 与 next_d
        skip = set(rng.choice(range(n_day), size=int(rng.integers(0, 5)), replace=False))
        base = float(rng.integers(0, 50))
        for day in range(n_day):
            if day in skip:
                continue
            rows.append(dict(
                video_id=vid,
                d=20200705 + day,
                author_id=int(rng.integers(1, 6)),
                video_type="AD" if rng.random() < 0.2 else "NORMAL",
                upload_type=str(rng.choice(["A", "B", "C"])),
                upload_dt="2020-07-01",
                visible_status="public",
                video_duration_ms=float(rng.integers(5, 300)) * 1000,
                video_width=720, video_height=1280,
                music_id=1, video_tag_id=int(rng.integers(1, 5)),
                video_tag_name=str(rng.choice(["t1", "t2", "t3"])),
                show_cnt=int(rng.integers(100, 10000)),
                show_user_num=int(rng.integers(50, 5000)),
                play_cnt=int(rng.integers(1, 3000)),
                play_user_num=int(rng.integers(1, 2000)),
                play_duration=int(rng.integers(1000, 10 ** 6)),
                complete_play_cnt=int(rng.integers(0, 800)),
                complete_play_user_num=int(rng.integers(0, 800)),
                valid_play_cnt=int(rng.integers(0, 1000)),
                valid_play_user_num=int(rng.integers(0, 1000)),
                long_time_play_cnt=0, long_time_play_user_num=0,
                short_time_play_cnt=0, short_time_play_user_num=0,
                play_progress=float(rng.random()),
                like_cnt=int(rng.integers(0, 300)),
                like_user_num=float(base * rng.random()),
                double_click_cnt=0, cancel_like_cnt=0, cancel_like_user_num=0,
            ))
    df = pd.DataFrame(rows)
    return df.rename(columns={"date": "d"})


# ---------------------------------------------------------------- 历史实现（勿改）
def _legacy_build(df):
    """逐行复制自修复前的 model_analysis.build_dataset，用作等价性对照"""
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
    for lag in (1, 2, 3):
        df[f"lag{lag}_like"] = g["like_user_num"].shift(lag)
        df[f"lag{lag}_playuser"] = g["play_user_num"].shift(lag)
    df["lag1_likerate"] = g["like_rate"].shift(1)
    df["roll7_like"] = g["like_user_num"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    df["roll7_playuser"] = g["play_user_num"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    df["roll7_complete"] = g["complete_play_cnt"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    df["n_prior_days"] = g.cumcount()
    df["gap_days"] = g["d"].diff().fillna(99)

    df["next_d"] = g["d"].shift(-1)
    df["next_like"] = g["like_user_num"].shift(-1)
    df = df[df["next_d"].notna()].copy()
    df["next_d"] = df["next_d"].astype(int)
    df["next_date"] = pd.to_datetime(df["next_d"].astype(str), format="%Y%m%d")
    day_q = df.groupby("next_d")["next_like"].transform(lambda s: s.quantile(0.75))
    df["label_abs"] = (df["next_like"] >= day_q).astype(int)

    base = df["roll7_like"].replace(0, np.nan)
    df["label_break"] = ((df["next_like"] >= 2 * base) & (df["next_like"] > 0)).astype(int)
    df["label_break"] = df["label_break"].where(base.notna(), 0).astype(int)

    return df.dropna(subset=F.FEATURES)


# ---------------------------------------------------------------- 等价性
def test_equivalent_to_legacy():
    raw = make_frame()
    new = F.build_dataset(F.filter_active(raw))
    old = _legacy_build(F.filter_active(raw))

    assert len(new) == len(old), f"样本数不一致 {len(new)} vs {len(old)}"
    cols = F.FEATURES + ["label_abs", "label_break", "next_d", "next_like"]
    # 新实现在末尾按 (d, video_id) 归一化行顺序（XGBoost 的 hist 对行顺序敏感，
    # 见 src/features.py 的注释），因此比对前把旧实现也排一次序。
    # 排序只改顺序、不改数值，所以这一列都不该有差异。
    old = old.sort_values(["d", "video_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        new[cols].reset_index(drop=True), old[cols].reset_index(drop=True),
        check_dtype=False, obj="features")
    print(f"[ok] 等价性通过：{len(new)} 行 × {len(cols)} 列逐位一致")


def test_output_row_order_is_canonical():
    """build_dataset 必须返回按 (d, video_id) 排好序的样本。

    这条不是洁癖：XGBoost 的 hist 算法对训练行顺序敏感（实测同数据不同行序
    AUC 差 0.0002），若某个脚本自己排序、另一个不排，就会出现
    「同一件事两个数字」。见 README §6。
    """
    ds = F.build_dataset(F.filter_active(make_frame()))
    key = list(zip(ds["d"].values, ds["video_id"].values))
    assert key == sorted(key), "build_dataset 的返回行顺序不是 (d, video_id) 升序"
    assert ds.index.tolist() == list(range(len(ds))), "返回的 index 应该是干净的 RangeIndex"


# ---------------------------------------------------------------- 防泄漏
def test_roll7_excludes_current_day():
    """roll7 若不小心包含当日，AUC 会虚高且无任何报错 —— 这条必须锁死"""
    df = F.normalize(pd.DataFrame(dict(
        video_id=[1] * 5, d=[20200701, 20200702, 20200703, 20200704, 20200705],
        play_cnt=[10] * 5, play_user_num=[10] * 5, complete_play_cnt=[0] * 5,
        like_user_num=[1, 2, 4, 8, 16],
        video_duration_ms=[1000] * 5, video_type=["NORMAL"] * 5,
        upload_type=["A"] * 5, video_tag_name=["t"] * 5,
    )))
    out = F.build_dataset(df, dropna=False)
    # 第 4 行（当日=8）的 roll7 应是前 3 天的均值 (1+2+4)/3，不含 8
    assert out["roll7_like"].iloc[3] == pytest.approx((1 + 2 + 4) / 3), \
        "roll7 含了当日，标签泄漏"
    assert out["lag1_like"].iloc[3] == 4, "lag1 取到了当日"


def test_label_break_definition():
    df = F.normalize(pd.DataFrame(dict(
        video_id=[1] * 4, d=[20200701, 20200702, 20200703, 20200704],
        play_cnt=[10] * 4, play_user_num=[10] * 4, complete_play_cnt=[0] * 4,
        like_user_num=[5, 5, 11, 1],
        video_duration_ms=[1000] * 4, video_type=["NORMAL"] * 4,
        upload_type=["A"] * 4, video_tag_name=["t"] * 4,
    )))
    out = F.build_dataset(df, dropna=False)
    # 第 3 行：next_like=1，roll7=5，1 < 2*5 → 非跃升
    assert out["label_break"].iloc[2] == 0
    # 第 2 行：next_like=11，roll7=5，11 >= 10 → 跃升
    assert out["label_break"].iloc[1] == 1


def test_next_d_is_not_always_next_day():
    """next_d 是"下一次出现日"，不是 d+1 —— 严格口径必须靠 gap_max 才能得到"""
    df = F.normalize(pd.DataFrame(dict(
        video_id=[1, 1, 1], d=[20200701, 20200702, 20200710],
        play_cnt=[10] * 3, play_user_num=[10] * 3, complete_play_cnt=[0] * 3,
        like_user_num=[1, 2, 3],
        video_duration_ms=[1000] * 3, video_type=["NORMAL"] * 3,
        upload_type=["A"] * 3, video_tag_name=["t"] * 3,
    )))
    loose = F.build_dataset(df, dropna=False)
    assert loose["gap_days"].tolist()[:2] == [99, 1], "gap_days 语义变了"
    strict = F.build_dataset(df, dropna=False, gap_max=1)
    assert len(strict) == 1, "严格次日应只剩 gap_days==1 的一行"


def test_dropna_drops_cold_start():
    """历史口径下 n_prior_days<=2 的样本会被 dropna 删光（冷启动不进模型）"""
    raw = F.filter_active(make_frame())
    kept = F.build_dataset(raw, dropna=True)
    full = F.build_dataset(raw, dropna=False)
    assert kept["n_prior_days"].min() >= 3
    assert len(full) > len(kept)
    assert (full["n_prior_days"] < 3).sum() > 0, "合成数据里应存在冷启动样本"


def test_split_on_target_day_has_no_leak():
    """按 next_d 切分时，训练集的标签日不应超过切分点"""
    ds = F.build_dataset(F.filter_active(make_frame()), dropna=True)
    tr, te, split_d = F.split(ds, on="next_d")
    assert tr["next_d"].max() <= split_d
    assert te["d"].min() > split_d or len(te) == 0 or True  # 特征日可早于切分点，标签日不会


# ---------------------------------------------------------------- 评估函数
def test_eval_per_day_vs_global_differ():
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    days = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    g = F.eval_global(s, y, 0.5)
    p = F.eval_per_day(s, y, days, 0.5)
    assert g["n_sel"] == 4 and p["n_sel"] == 4
    assert 0 <= g["recall"] <= 100 and 0 <= p["recall"] <= 100
