# outputs/ —— 每个数字的出处

本目录是 README 所有数字的唯一来源。**任何在这里查不到的数字，都不该出现在 README 里。**
每次改动脚本后重跑，请一并更新下表；表中「README 引用位置」为空的行，说明该文件目前没有
被正文引用，属于历史产物。

## 现行产物（可以引用）

| 文件 | 生成脚本 | 内容 | README 引用位置 |
|---|---|---|---|
| `base_metrics.json` | `src/run_analysis.py` | 13 组业务指标（漏斗/完播分布/广告对比/时长分箱/时段/分层） | §2、§4、业务洞察图 |
| `exp_eval_upgrade.json` | `experiments/eval_upgrade.py` | **口径主表**：两种标签口径 × LR/XGB/基线 × AUC/AP × global/per-day × top5–30 | 总表、§3.3、§4.1、§4.2、§4.3 全部 top-k 数字 |
| `exp_ablation.json` | `experiments/ablation.py` | 8 组特征消融 × 2 标签 × 2 模型，含 roll7 分母效应的量化（G7→G8） | §4.1（0.9785 / 0.9813 / +0.0028）、§4.4（0.5310 / 0.8085 / 0.6055 / −0.0005） |
| `label_robustness.csv` | ⚠ **仓库内没有对应脚本** | 三种跃升标签定义的稳健性（2 倍阈值 / 最小量 K=20 / 泊松 +2σ） | §5 局限（说明阈值敏感性已做） |
| `exp_significance.json` | `experiments/significance.py` | 按天 bootstrap 的差值 95%CI + 5 个种子的均值±标准差 | §4.1「无增量」的统计依据（首次生成需重跑） |

## 历史产物（**不要引用**）

| 文件 | 状态 | 说明 |
|---|---|---|
| `baseline_vs_model.json` | 旧口径：全局混排 | top20 召回 76.17% / 76.39%，与 per-day 的 77.01% / 77.22% **不是同一组数字**。正文一律引用 `exp_eval_upgrade.json`。保留它只为按 `n_prior_days` 分段拆增益，以及留痕「先立基线」这一步 |
| `breakout_experiment.json` | 早期口径：全局混排 | 结论方向可用，数字勿与 per-day 混引。该脚本现在额外输出「方向检验」三条排序线，用于检验 AUC 0.43 的解释 |
| `model_metrics.json` | **旧版脚本产物，字段已与当前 `src/model_analysis.py` 不一致** | 里面的 `auc_xgb = 0.9825`、`recall_xgb/precision_xgb` 是旧版脚本写的；当前脚本产出的是 `topk_recall` 字典与 `importance`。仓库里的这一份**不是** 0.9813 那次运行的结果。重跑 `python src/model_analysis.py` 会原地覆盖 |
| `metrics.json` | 历史合并格式（`base` / `model` / `project1` / `bullets`） | 早期把 SQL 与模型结果合在一个文件里的版本，已被 `base_metrics.json` + `model_metrics.json` 取代。仅 `experiments/baseline_check.py` 的注释提到它 |

## 关于「三个不同的模型 AUC」

如果你在仓库里看到 0.9825 / 0.9813 / 0.9811 三个数，它们分别是：

- **0.9825** —— `model_metrics.json`，旧版 `model_analysis.py` 产物（口径未记录，已不可复现）
- **0.9813** —— `exp_eval_upgrade.json`，当前 `eval_upgrade.py`，**README 采用这一个**
- **0.9811** —— `baseline_vs_model.json`，当前 `baseline_vs_model.py`，与上一行的差异来自
  全局混排 vs per-day 两种评估口径下测试集构成的细微不同 + 多线程非确定性

0.9813 与 0.9811 的差正好落在种子抖动区间（0.0005–0.0021）内，这也说明一个问题：
**跨脚本引用数字时必须注明出处，否则两个「同义」数字会互相证伪。**

## 复现顺序

```bash
python run_all.py --with-experiments     # base_metrics / model_metrics / exp_eval_upgrade / exp_ablation / exp_significance
python experiments/breakout_experiment.py   # 早期口径 + 方向检验
python experiments/baseline_vs_model.py     # 旧口径分段（勿混引）
python experiments/plot_ablation.py         # 只读 exp_ablation.json 重出图，不必重跑模型
```
