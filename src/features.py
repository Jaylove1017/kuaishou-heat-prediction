# -*- coding: utf-8 -*-
"""全项目唯一的特征工程实现 —— 单一事实来源（single source of truth）

为什么要抽这一层
----------------
历史上这份逻辑在五个地方各存了一份副本：
    src/model_analysis.py、experiments/baseline_vs_model.py、
    experiments/breakout_experiment.py、experiments/eval_upgrade.py、
    experiments/ablation.py
它们已经开始漂移：model_analysis 从 MySQL 读，其余从 data/clean/*.csv 读
（两条数据通路）；n_jobs 一处 -1、一处 4；fillna 写法也不统一。
而本项目全部结论都建立在「跨脚本对比同一件事」上，口径一旦分叉，
任何两个数字之间都不再保证可比。所以统一到这里，其余脚本只 import。

口径约定（改之前务必读完）
--------------------------
* `roll7_*` 与 `lag*_*` 一律 shift(1) 之后再滚动/取值 —— **绝不含当日**。
  这是全项目最关键的防泄漏约定，tests/test_features.py 会把它锁死。
* 标签 A `label_abs`：next_like ≥ 当日（按 next_d 分组）可预测样本的 75 分位。
  注意分母是「当日有下一观测日的视频」，不是「当日全部活跃视频」。
* 标签 B `label_break`：next_like ≥ break_mult × roll7_like，且 next_like > 0。
  roll7_like 在这里既是标签分母、又是特征之一，这层重叠的量化见
  experiments/ablation.py 的 G7 → G8。
* `next_d` 是「该视频下一次出现在 video_daily 的日期」，**不一定是 d+1**。
  需要严格次日口径时传 gap_max=1（按 `lead_days = next_d - d` 过滤）。
  注意 `gap_days`（向后，距上次观测）与 `lead_days`（向前，距标签日）是两个量，
  前者是特征、后者决定标签窗口长度，不要混用。

兼容性说明
----------
所有默认参数都保持与历史脚本完全一致，因此默认路径下样本数、切分点、
标签分布与之前提交的 outputs/*.json 逐位对齐（见 tests/test_features.py 的等价性用例）。
更严格的口径（严格次日、按目标日切分、保留缺失值）通过参数或命令行开关开启，
默认关闭，避免静默改变已发布的数字。
"""
import os

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- 全局常量（所有脚本共用，不要在别处重新定义）----
SEED = 42
N_JOBS = int(os.environ.get("N_JOBS", "4"))   # 统一并发数，避免各脚本抖动区间不同
SPLIT_TAIL_DAYS = 14                          # 末尾 N 天做测试集
TOP_Q = 0.75                                  # 高潜 = 次日点赞 ≥ 当日前 25%
MIN_DAY = 20200705                            # 数据起始日（yyyyMMdd）
BREAK_MULT = 2.0                              # 跃升阈值：自身近 7 日均值的倍数
ROLL_WINDOW = 7
LAGS = (1, 2, 3)

# ---- 工业化的 22 维特征 ----（消融实验按这三组拆）
CONTENT = ["dur_log", "is_ad", "up_type_idx", "tag_idx", "dow"]
HEAT = ["play_user_num", "like_user_num", "complete_play_cnt", "like_rate", "complete_rate"]
INERTIA = ["lag1_like", "lag2_like", "lag3_like",
           "lag1_playuser", "lag2_playuser", "lag3_playuser", "lag1_likerate",
           "roll7_like", "roll7_playuser", "roll7_complete", "n_prior_days", "gap_days"]
FEATURES = CONTENT + HEAT + INERTIA

assert len(CONTENT) == 5 and len(HEAT) == 5 and len(INERTIA) == 12
assert len(set(FEATURES)) == 22, "特征重复"


# ---------------------------------------------------------------- 数据读取
def normalize(df):
    """统一 CSV 与 MySQL 两条通路的列名差异，返回可直接喂给 build_dataset 的表"""
    df = df.rename(columns={"date": "d", "video_duration": "video_duration_ms"})
    missing = {"d", "video_id", "play_cnt", "like_user_num", "play_user_num"} - set(df.columns)
    if missing:
        raise KeyError(f"缺少必需列: {sorted(missing)}")
    return df


def load_daily_csv(path=None):
    """从 data/clean/video_daily.csv 读内容日维表（experiments/ 统一入口）"""
    path = path or os.path.join(PROJ, "data", "clean", "video_daily.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到 {path}\n"
            f"请先按 data/README.md 下载原始数据，再运行 python src/load_data.py 生成清洗文件。"
        )
    return normalize(pd.read_csv(path, low_memory=False))


def filter_active(df):
    """只保留有播放且落在数据窗口内的视频日（与 model_analysis.py 的 SQL WHERE 一致）"""
    return df[(df["play_cnt"] > 0) & (df["d"] >= MIN_DAY)].copy()


# ---------------------------------------------------------------- 特征 + 标签
def build_dataset(df, *, dropna=True, gap_max=None,
                  break_mult=BREAK_MULT, top_q=TOP_Q, with_break=True):
    """构造 22 维特征与两种标签。

    参数
    ----
    dropna     : True（默认，历史口径）= 按 FEATURES 全集 dropna；False = 保留缺失值交给 XGBoost
    gap_max    : None（默认，历史口径）= 不限间隔；1 = 严格次日（next_d == d+1，按 lead_days 过滤）
    break_mult : 跃升阈值倍数，默认 2.0（可做 1.5 / 3.0 敏感性）
    top_q      : 标签 A 的分位点，默认 0.75
    with_break : 是否构造跃升标签
    """
    df = df.copy()

    df["date"] = pd.to_datetime(df["d"].astype(str), format="%Y%m%d")
    df["dur_log"] = np.log1p(df["video_duration_ms"].fillna(0) / 1000.0)
    df["is_ad"] = (df["video_type"] == "AD").astype(int)
    df["up_type_idx"] = df["upload_type"].fillna("UNKNOWN").astype("category").cat.codes
    df["tag_idx"] = df["video_tag_name"].fillna("_NA_").astype("category").cat.codes
    # day_of_week 与 weekday 等价，但前者在 pandas 4 里才不会被 remove
    df["dow"] = df["date"].dt.day_of_week
    df["like_rate"] = (df["like_user_num"] / df["play_user_num"].replace(0, np.nan)).fillna(0)
    df["complete_rate"] = (df["complete_play_cnt"] / df["play_cnt"].replace(0, np.nan)).fillna(0)

    g = df.sort_values("d").groupby("video_id")
    # t-1 / t-2 / t-3 滞后：shift 从 1 开始，当日不进特征
    for lag in LAGS:
        df[f"lag{lag}_like"] = g["like_user_num"].shift(lag)
        df[f"lag{lag}_playuser"] = g["play_user_num"].shift(lag)
    df["lag1_likerate"] = g["like_rate"].shift(1)
    # 7 日滚动均值：同样先 shift(1)，**不含当日**
    for col, name in (("like_user_num", "roll7_like"),
                      ("play_user_num", "roll7_playuser"),
                      ("complete_play_cnt", "roll7_complete")):
        df[name] = g[col].transform(
            lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean())
    df["n_prior_days"] = g.cumcount()
    df["gap_days"] = g["d"].diff().fillna(99)

    # 目标：下一次出现日的点赞量
    df["next_d"] = g["d"].shift(-1)
    df["next_like"] = g["like_user_num"].shift(-1)
    df = df[df["next_d"].notna()].copy()
    df["next_d"] = df["next_d"].astype(int)
    df["next_date"] = pd.to_datetime(df["next_d"].astype(str), format="%Y%m%d")
    # **向前**间隔：从今天到标签日有多远。这才是「次日」这个词对应的量。
    # 注意不要和上面的 gap_days 混用：gap_days 是「距离上一次观测多久」（向后），
    # 它是合法特征（视频是否在持续更新），但不能用来判断标签窗口长度。
    df["lead_days"] = df["next_d"] - df["d"]

    # 严格次日口径：只保留 next_d == d+1 的样本（在算标签之前过滤，
    # 这样标签分位数也只在严格次日样本内部取，口径才自洽）
    if gap_max is not None:
        df = df[df["lead_days"] <= gap_max].copy()
        if df.empty:
            raise ValueError(f"gap_max={gap_max} 过滤后没有样本，请检查数据")

    # 标签 A：次日点赞是否达到当日可预测样本的前 25%
    day_q = df.groupby("next_d")["next_like"].transform(lambda s: s.quantile(top_q))
    df["label_abs"] = (df["next_like"] >= day_q).astype(int)

    # 标签 B：相对自身近 7 日均值的跃升
    if with_break:
        base = df["roll7_like"].replace(0, np.nan)
        df["label_break"] = ((df["next_like"] >= break_mult * base) & (df["next_like"] > 0)).astype(int)
        df["label_break"] = df["label_break"].where(base.notna(), 0).astype(int)

    if dropna:
        df = df.dropna(subset=FEATURES).copy()

    # 行顺序统一在这里定一次：XGBoost 的 hist 算法对行顺序有 1e-4 级敏感
    # （实测同一份数据、同一组超参，仅改行顺序 AUC 就从 0.980863 变到 0.981078）。
    # 历史上 eval_upgrade 排了序、baseline_vs_model 没排，两个脚本因此给出
    # 0.9811 / 0.9809 的"同一件事"两个数字。统一到此处，任何消费方拿到的顺序都相同。
    return df.sort_values(["d", "video_id"]).reset_index(drop=True)


def split(ds, tail_days=SPLIT_TAIL_DAYS, on="d"):
    """时间切分。返回 (train, test, split_day)

    on="d"       （默认，历史口径）：按特征日切分
    on="next_d"  （严格口径）：按目标日切分，保证训练集里没有标签落在测试窗口的样本
    """
    if on not in ("d", "next_d"):
        raise ValueError("on 只能是 'd' 或 'next_d'")
    day_max = ds["d"].max()
    split_d = int((pd.to_datetime(str(day_max), format="%Y%m%d")
                   - pd.Timedelta(days=tail_days)).strftime("%Y%m%d"))
    key = ds[on]
    tr, te = ds[key <= split_d], ds[key > split_d]
    return tr, te, split_d


# ---------------------------------------------------------------- 模型工厂
def make_xgb(seed=SEED, n_jobs=None):
    """统一超参。历史上 n_jobs 不统一（-1 / 4），会让不同脚本的抖动区间不可比。"""
    import xgboost as xgb
    return xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.06, max_depth=6,
        subsample=0.9, colsample_bytree=0.9, eval_metric="auc",
        tree_method="hist", n_jobs=N_JOBS if n_jobs is None else n_jobs,
        random_state=seed, importance_type="gain",
    )


def make_lr(seed=SEED):
    """LR 流水线。

    包含中位数填补：XGBoost 原生接受缺失值，但 sklearn 的 LogisticRegression 不接受，
    所以 model_analysis.py --keep-na（保留冷启动样本）时 LR 这条线会直接抛
    "Input X contains NaN"。填补放在 pipeline 里，全量样本时退化为恒等变换，
    不影响原有数值。
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median"),
                         StandardScaler(),
                         LogisticRegression(max_iter=3000, C=0.5,
                                            class_weight="balanced", random_state=seed))


# ---------------------------------------------------------------- 评估
def eval_global(scores, y, k):
    """全局混排：把全部测试样本放在一起取 top-k%（旧口径，业务上不对应任何动作）"""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y)
    n = max(int(len(y) * k), 1)
    idx = np.argsort(-scores)[:n]
    hits = int(y[idx].sum())
    return dict(recall=hits / max(int(y.sum()), 1) * 100,
                precision=hits / n * 100, n_sel=n, hits=hits)


def eval_per_day(scores, y, days, k):
    """按天分别圈选再汇总（业务正确口径：每天收盘圈一次明天的候选）"""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y)
    days = np.asarray(days)
    hits = n_sel = 0
    for d in np.unique(days):
        m = days == d
        s, yy = scores[m], y[m]
        if len(s) == 0:
            continue
        n = max(int(m.sum() * k), 1)
        idx = np.argsort(-s)[:n]
        hits += int(yy[idx].sum())
        n_sel += n
    return dict(recall=hits / max(int(y.sum()), 1) * 100,
                precision=hits / max(n_sel, 1) * 100, n_sel=n_sel, hits=hits)


# ---------------------------------------------------------------- 绘图
def setup_style():
    """中文字体跨平台回退（原实现硬编码 Microsoft YaHei，Linux/macOS 上出图全是方块）"""
    import matplotlib
    from matplotlib import font_manager
    pref = ["Microsoft YaHei", "SimHei", "PingFang SC", "Heiti SC",
            "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Micro Hei"]
    have = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [n for n in pref if n in have]
    if not chosen:
        print("[warn] 未找到中文字体，图中中文可能显示为方块；"
              "Linux 可 apt install fonts-wqy-microhei")
        chosen = []
    matplotlib.rcParams["font.sans-serif"] = chosen + ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return chosen
