-- ============================================================
-- KuaiRec 业务指标分析 SQL（结果由 src/run_analysis.py 抓取归档）
-- 分区说明见 src/run_analysis.py 的 marker 映射
-- ============================================================

-- Q: 数据规模与质量
SELECT COUNT(*) AS play_rows,
       COUNT(DISTINCT user_id) AS users,
       COUNT(DISTINCT video_id) AS videos,
       COUNT(DISTINCT d) AS days,
       SUM(is_valid=1) AS valid_rows,
       SUM(is_valid=0) AS abnormal_rows
FROM play_log;

-- Q: 完播分布（有效播放）
SELECT
  SUM(watch_ratio >= 1.0) / COUNT(*) * 100   AS ratio_full_pct,   -- 完播(≥1x)
  SUM(watch_ratio >= 0.9) / COUNT(*) * 100   AS ratio_90_pct,
  SUM(watch_ratio <  0.2) / COUNT(*) * 100   AS ratio_skip_pct,   -- 快划跳过(<0.2)
  AVG(watch_ratio)                           AS ratio_avg,
  SUM(watch_ratio) / COUNT(*) * 100          AS ratio_sum_x
FROM play_log WHERE is_valid = 1;

-- Q: 内容侧漏斗（video_daily 全期合计）
SELECT
  SUM(show_cnt)              AS show_cnt,
  SUM(play_cnt)              AS play_cnt,
  SUM(valid_play_cnt)        AS valid_cnt,
  SUM(complete_play_cnt)     AS complete_cnt,
  SUM(like_cnt)              AS like_cnt,
  SUM(show_user_num)         AS show_user,
  SUM(play_user_num)         AS play_user,
  SUM(valid_play_user_num)   AS valid_user,
  SUM(complete_play_user_num) AS complete_user,
  SUM(like_user_num)         AS like_user,
  COUNT(*)                   AS video_days
FROM video_daily;

-- Q: 完播口径对照（播放侧 vs 内容侧）
SELECT
  SUM(complete_play_cnt) / NULLIF(SUM(play_cnt), 0) * 100  AS complete_of_play_pct,
  SUM(valid_play_cnt)  / NULLIF(SUM(play_cnt), 0) * 100    AS valid_of_play_pct,
  SUM(like_cnt)        / NULLIF(SUM(play_cnt), 0) * 100    AS like_rate_per_play_pct
FROM video_daily;

-- Q: 头部视频长尾（按全期 play_cnt 排序，头部 1%/5% 视频贡献份额）
-- 先用中间表思路：全期每视频聚合
WITH per_video AS (
  SELECT video_id,
         SUM(play_cnt) AS play,
         SUM(play_user_num) AS play_user,
         SUM(complete_play_cnt) AS complete_cnt,
         SUM(like_cnt) AS like_cnt,
         SUM(show_cnt) AS show_cnt
  FROM video_daily
  GROUP BY video_id
)
SELECT
  (SELECT COUNT(*) FROM per_video) AS n_videos,
  ROUND((SELECT SUM(play) FROM (SELECT play FROM per_video ORDER BY play DESC LIMIT 107) AS t1) / NULLIF((SELECT SUM(play) FROM per_video), 0) * 100, 2) AS top1pct_play_share,
  ROUND((SELECT SUM(like_cnt) FROM (SELECT like_cnt FROM per_video ORDER BY like_cnt DESC LIMIT 107) AS t2) / NULLIF((SELECT SUM(like_cnt) FROM per_video), 0) * 100, 2) AS top1pct_like_share
FROM per_video LIMIT 1;

-- Q: 点赞正反馈长尾（like_cnt>0 的视频）
SELECT
  ROUND(SUM(like_cnt > 0) / COUNT(*) * 100, 2) AS video_days_with_like_pct
FROM video_daily;

-- Q: 广告 vs 普通内容互动
SELECT video_type,
       COUNT(*) AS video_days,
       ROUND(AVG(play_cnt), 1) AS avg_play,
       ROUND(SUM(complete_play_cnt) / NULLIF(SUM(play_cnt), 0) * 100, 2) AS complete_pct,
       ROUND(SUM(like_cnt) / NULLIF(SUM(play_user_num), 0) * 100, 3) AS like_per_playuser_pct
FROM video_daily
GROUP BY video_type
ORDER BY avg_play DESC;

-- Q: 时长对完播影响（把 video_duration_ms 分箱：<30s / 30-60s / 1-3min / 3-10min / >10min）
SELECT
  CASE
    WHEN video_duration_ms < 30000 THEN '<30s'
    WHEN video_duration_ms < 60000 THEN '30-60s'
    WHEN video_duration_ms < 180000 THEN '1-3min'
    WHEN video_duration_ms < 600000 THEN '3-10min'
    ELSE '>10min'
  END AS dur_bucket,
  COUNT(*) AS video_days,
  ROUND(AVG(video_duration_ms)/1000, 1) AS avg_sec,
  ROUND(SUM(complete_play_cnt) / NULLIF(SUM(play_cnt), 0) * 100, 2) AS complete_pct,
  ROUND(SUM(like_cnt) / NULLIF(SUM(play_user_num), 0) * 100, 3) AS like_per_playuser_pct
FROM video_daily
GROUP BY dur_bucket
ORDER BY MIN(video_duration_ms);

-- Q: 分日播放与活跃（趋势）
SELECT d,
       SUM(play_cnt) AS play_cnt,
       SUM(like_cnt) AS like_cnt
FROM video_daily
GROUP BY d
ORDER BY d;

-- Q: 时段活跃（播放侧 big_matrix，北京时间小时）
SELECT hour,
       COUNT(*) AS plays,
       SUM(watch_ratio >= 1.0) AS complete_plays
FROM play_log
WHERE is_valid = 1
GROUP BY hour
ORDER BY hour;

-- Q: 用户侧：人均播放 / 活跃天数 / 完播比（有效播放）
SELECT COUNT(*) / COUNT(DISTINCT user_id) AS plays_per_user,
       COUNT(DISTINCT d) AS total_days
FROM play_log
WHERE is_valid = 1;

-- Q: 用户完播率分布（每用户 ratio>=1 占比），用于分层
SELECT
  CASE
    WHEN u.full_pct >= 40 THEN 'high_completer(>=40%)'
    WHEN u.full_pct >= 20 THEN 'mid_completer(20-40%)'
    ELSE 'light_completer(<20%)'
  END AS seg,
  COUNT(*) AS users,
  ROUND(AVG(u.plays), 0) AS avg_plays
FROM (
  SELECT user_id,
         SUM(watch_ratio >= 1.0) / COUNT(*) * 100 AS full_pct,
         COUNT(*) AS plays
  FROM play_log WHERE is_valid = 1
  GROUP BY user_id
) u
GROUP BY seg
ORDER BY users DESC;

-- Q: 视频日维高潜候选量级：次日可预测对象（次日播放>0 且有 lag 可用 的视频日数）
SELECT COUNT(*) AS rows_for_model
FROM video_daily a
WHERE EXISTS (SELECT 1 FROM video_daily b WHERE b.video_id = a.video_id AND b.d < a.d)
  AND a.play_cnt > 0;
