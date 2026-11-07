# outputs/ —— 每个数字的出处

本目录是 README 所有数字的唯一来源。**任何在这里查不到的数字，都不该出现在 README 里。**
每次改动脚本后重跑，请一并更新下表；表中「README 引用位置」为空的行，说明该文件目前没有
被正文引用，属于历史产物。

## 现行产物（可以引用）

| 文件 | 生成脚本 | 内容 | README 引用位置 |
|---|---|---|---|
| `base_metrics.json` | `src/run_analysis.py` | 13 组业务指标（漏斗/完播分布/广告对比/时长分箱/时段/分层）与数据规模 | §2 |
| `exp_eval_upgrade.json` | `experiments/eval_upgrade.py` | **口径主表**：两种标签口径 × LR/XGB/基线 × AUC/AP × global/per-day × top5–30 | 总表、§3.1、§3.3、§4.1、§4.2 |
| `exp_significance.json` | `experiments/significance.py` | 1000 次按天配对 bootstrap 的差值 95%CI + 5 个种子的均值±标准差 | §3.1 的 CI 表、§3.4、§4.2 |
| `exp_ablation.json` | `experiments/ablation.py` | 8 组特征消融 × 2 标签 × 2 模型，含 roll7 分母效应的量化（G7→G8） | §3.2、§4.1、§4.4 |
| `exp_strict.json` | `experiments/strict_eval.py` | 四种口径（严格次日 / 保留冷启动 / 按目标日切分）× gap 分布 × 分段表现 | §4.5 全部数字 |
| `exp_playlog.json` | `experiments/playlog_features.py` | play_log 分布特征的四组对照（含覆盖子集 AUC） | §4.6 全部数字 |
| `breakout_experiment.json` | `experiments/breakout_experiment.py` | 跃升标签早期口径 + **方向检验**三条排序线 | §4.3 的方向检验表 |
| `model_metrics.json` | `src/model_analysis.py` | 单脚本视角的 22 维结果 + top-k 召回 + gain 重要性 | 与主表同源，用于交叉验证 |
| `label_robustness.csv` | ⚠ **仓库内没有对应脚本** | 三种跃升标签定义的稳健性（2 倍阈值 / 最小量 K=20 / 泊松 +2σ） | §5 局限 |

## 历史产物（**不要引用**）

| 文件 | 状态 | 说明 |
|---|---|---|
| `baseline_vs_model.json` | 旧口径：全局混排 | top20 召回 76.28% / 76.39%，与 per-day 的 77.05% / 77.21% **不是同一组数字**。正文一律引用 `exp_eval_upgrade.json`。保留它只为按 `n_prior_days` 分段拆增益 |
| `metrics.json` | 历史合并格式（`base` / `model` / `project1` / `bullets`） | 早期把 SQL 与模型结果合在一个文件里的版本，已被 `base_metrics.json` + `model_metrics.json` 取代。仅 `experiments/baseline_check.py` 的注释提到它 |

## 关于「模型 AUC 到底是 0.9813 还是 0.9811」

如果你在 git 历史或旧材料里看到 **0.9825 / 0.9813 / 0.9811** 三个数，现在只有 **0.9811** 有效：

| 数字 | 来源 | 现状 |
|---|---|---|
| 0.9825 | 旧版 `model_metrics.json` | 旧脚本口径（未过滤 `play_cnt > 0`，样本 302,019），已不可复现 |
| 0.9813 | 旧版 `exp_eval_upgrade.json` | 旧 xgboost 版本下的结果，同代码重跑得 0.9811 |
| **0.9811** | 当前全部脚本 | **唯一有效值**，`eval_upgrade` / `baseline_vs_model` / `model_analysis` / `strict_eval` 四者一致 |

为什么会有 0.0002 的漂移，查清楚了两件事，都写进了 README §6：

1. **库版本**：早期 json 由较老的 xgboost 生成。`requirements.txt` 现在锁定主版本区间，
   跨大版本升级必须重跑并同步 README。
2. **行顺序**：`hist` 算法对训练行顺序敏感。实测同一份数据仅改行顺序，
   AUC 从 0.980863 变到 0.981078。历史上 `eval_upgrade.py` 排了序、`baseline_vs_model.py` 没排，
   于是「同一件事」出现两个数字。现在排序统一在 `src/features.py` 的 `build_dataset()` 末尾做一次。

这两条合起来说明：**跨脚本引用数字时必须注明出处，否则两个「同义」数字会互相证伪。**

## 复现顺序

```bash
python run_all.py --with-experiments       # base_metrics / model_metrics / exp_eval_upgrade / exp_ablation / exp_significance
python experiments/strict_eval.py          # 四种口径审计
python experiments/playlog_features.py     # 播放日志特征（负结果）
python experiments/breakout_experiment.py  # 跃升标签 + 方向检验
python experiments/baseline_vs_model.py    # 旧口径分段（勿混引）
python experiments/plot_ablation.py        # 只读 exp_ablation.json 重出图，不必重跑模型
```

除 `run_analysis.py` 需要 MySQL 外，其余都只读 `data/clean/*.csv`。
