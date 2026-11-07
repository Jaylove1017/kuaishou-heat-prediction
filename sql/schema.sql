-- KuaiRec（快手官方公开论文数据）分析数仓 schema
-- 说明：play_log 来自 big_matrix.csv.gz（播放行为流），video_daily 来自 item_daily_features.csv.gz（内容×日平台统计）
--
-- ⚠ 列顺序与 src/load_data.py 的 VD_COLS / PLAY_COLS 必须一致
--   LOAD DATA 现在显式带列名（(...) 子句），不再依赖位置对齐；但表里多一列少一列
--   仍然会让下游 SQL 拿到 NULL，所以改动任何一侧都要同步另一侧。
--   历史上这里多了一个 click_like_cnt，而清洗脚本的 usecols 少写了它，
--   导致第 30 列起整体错位一列且不报错 —— 详见 src/load_data.py 头部注释。
CREATE DATABASE IF NOT EXISTS kuairec CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE kuairec;

DROP TABLE IF EXISTS play_log;
CREATE TABLE play_log (
    user_id          INT          NOT NULL,
    video_id         INT          NOT NULL,
    play_duration_ms BIGINT       NULL,
    video_duration_ms BIGINT      NULL,
    d                INT          NOT NULL COMMENT 'yyyyMMdd',
    hour             TINYINT      NOT NULL COMMENT '播放发生的小时 0-23',
    watch_ratio      DOUBLE       NULL COMMENT 'play_duration/video_duration，可>1(重复观看)',
    is_valid         TINYINT      NOT NULL DEFAULT 1 COMMENT '有效播放：时长>0 且 watch_ratio<=10（剔除异常流）',
    KEY idx_video (video_id),
    KEY idx_user  (user_id),
    KEY idx_d     (d),
    KEY idx_valid (is_valid),
    -- 不建 UNIQUE：一次播放未必能由 (user_id,video_id,d,hour) 唯一确定
    -- （同小时内可能重复播放），加唯一键有丢数据风险。
    -- 改为建普通索引，供 load_data.py 的重复检测查询走索引，而不是全表扫。
    KEY idx_play_dedup (user_id, video_id, d, hour)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4; 

DROP TABLE IF EXISTS video_daily;
CREATE TABLE video_daily (
    video_id            INT          NOT NULL,
    d                   INT          NOT NULL COMMENT 'yyyyMMdd',
    author_id           INT          NOT NULL,
    video_type          VARCHAR(16)  NULL COMMENT 'NORMAL/AD',
    upload_type         VARCHAR(32)  NULL,
    upload_dt           VARCHAR(16)  NULL,
    visible_status      VARCHAR(16)  NULL,
    video_duration_ms   DOUBLE       NULL,
    video_width         INT          NULL,
    video_height        INT          NULL,
    music_id            BIGINT       NULL,
    video_tag_id        INT          NULL,
    video_tag_name      VARCHAR(64)  NULL COMMENT '画面内容标签',
    show_cnt            BIGINT       NULL,
    show_user_num       BIGINT       NULL,
    play_cnt            BIGINT       NULL,
    play_user_num       BIGINT       NULL,
    play_duration       BIGINT       NULL,
    complete_play_cnt   BIGINT       NULL,
    complete_play_user_num BIGINT    NULL,
    valid_play_cnt      BIGINT       NULL,
    valid_play_user_num BIGINT       NULL,
    long_time_play_cnt  BIGINT       NULL,
    long_time_play_user_num BIGINT   NULL,
    short_time_play_cnt BIGINT       NULL,
    short_time_play_user_num BIGINT  NULL,
    play_progress       DOUBLE       NULL,
    like_cnt            BIGINT       NULL,
    like_user_num       BIGINT       NULL,
    click_like_cnt      BIGINT       NULL,
    double_click_cnt    BIGINT       NULL,
    cancel_like_cnt     BIGINT       NULL,
    cancel_like_user_num BIGINT      NULL,
    KEY idx_video_daily_video (video_id, d),
    KEY idx_video_daily_author (author_id),
    KEY idx_video_daily_d (d)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
