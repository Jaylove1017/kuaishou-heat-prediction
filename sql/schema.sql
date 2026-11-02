-- KuaiRec（快手官方公开论文数据）分析数仓 schema
-- 说明：play_log 来自 big_matrix.csv.gz（播放行为流），video_daily 来自 item_daily_features.csv.gz（内容×日平台统计）
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
    KEY idx_valid (is_valid)
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
