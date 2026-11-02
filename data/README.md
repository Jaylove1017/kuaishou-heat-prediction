# data/ — 数据文件不随仓库分发

本目录的两个子文件夹在本仓库中是空的。原始与清洗后数据合计约 **930 MB**，不适合放进 Git。

## 需要下载什么

| 文件 | 放到 | 体积 | 说明 |
|---|---|---|---|
| `big_matrix.csv.gz` | `data/raw/` | ~329 MB | 全量曝光-播放日志，1,253 万条 |
| `item_daily_features.csv.gz` | `data/raw/` | ~21 MB | 内容×日统计，10,728 视频 × 63 天 |

来源：HuggingFace 数据集仓库 [`hiiamkik/kuai-rec-data`](https://huggingface.co/datasets/hiiamkik/kuai-rec-data)。
这是快手团队在 RecSys 2022 论文中公开的 **KuaiRec** 学术数据集，官方注明学术用途。

## 下载方式

```bash
# 方式一：huggingface-cli
pip install huggingface_hub
huggingface-cli download hiiamkik/kuai-rec-data --repo-type dataset \
  --include "big_matrix.csv.gz" "item_daily_features.csv.gz" \
  --local-dir data/raw

# 方式二：浏览器下载后手动放入 data/raw/
```

## 清洗后的文件（自动生成，不必下载）

运行 `python src/load_data.py` 后会生成：

- `data/clean/play_log.csv` —— 12,257,467 行（原始 1,253 万剔除 2.2%）
- `data/clean/video_daily.csv` —— 343,341 行

这两个文件同样被 `.gitignore` 排除，属于可复现的中间产物。
