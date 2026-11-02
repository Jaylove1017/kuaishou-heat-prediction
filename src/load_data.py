# -*- coding: utf-8 -*-
"""清洗 big_matrix / item_daily_features 并整库导入 MySQL kuairec 库
用法: python src/load_data.py [--skip-load]
"""
import argparse
import csv
import gzip
import io
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(PROJ, "data", "raw")
CLEAN = os.path.join(PROJ, "data", "clean")

# ---- MySQL 连接（默认本次验证实例；用户自跑可改）----
DB = dict(host=os.environ.get("MYSQL_HOST", "127.0.0.1"),
          port=int(os.environ.get("MYSQL_PORT", "3307")),
          user=os.environ.get("MYSQL_USER", "root"),
          password=os.environ.get("MYSQL_PASSWORD", ""),
          database=os.environ.get("MYSQL_DB", "kuairec"))


def get_conn(autocommit=True):
    import pymysql
    return pymysql.connect(host=DB["host"], port=DB["port"], user=DB["user"],
                           password=DB["password"], database=DB["database"],
                           autocommit=autocommit, local_infile=True, charset="utf8mb4")


def run_schema(cur):
    with open(os.path.join(PROJ, "sql", "schema.sql"), "r", encoding="utf-8") as f:
        sql = f.read()
    for stmt in sql.split(";"):
        stmt = stmt.strip()
        if stmt and not stmt.upper().startswith(("USE ", "CREATE DATABASE")):
            cur.execute(stmt)
    print("schema ok: play_log & video_daily 已建")


def clean_play_log():
    """流式清洗播放流 → data/clean/play_log.csv（明文，便于断点观察与 LOAD DATA）"""
    src = os.path.join(RAW, "big_matrix.csv.gz")
    dst = os.path.join(CLEAN, "play_log.csv")
    os.makedirs(CLEAN, exist_ok=True)
    total = valid = bad = 0
    t0 = time.time()
    out = open(dst, "w", encoding="utf-8", newline="", buffering=1024 * 1024)
    w = csv.writer(out)
    with gzip.open(src, "rt", encoding="utf-8", errors="replace", newline="") as f:
        rd = csv.reader(f)
        next(rd)
        for row in rd:
            total += 1
            if len(row) != 8:
                bad += 1
                continue
            try:
                uid = int(row[0]); vid = int(row[1])
                pdur = int(float(row[2])); vdur = int(float(row[3]))
                ts = float(row[6])
                ratio = float(row[7])
                d = int(row[5])
            except (ValueError, TypeError):
                bad += 1
                continue
            if pdur <= 0 or vdur <= 0:
                bad += 1
                continue
            # 数据时间字段为 UTC 壁钟；+8 换算北京时间用于“时段活跃”口径
            hour = int((ts % 86400) // 3600 + 8) % 24
            is_valid = 1 if ratio <= 10.0 else 0
            w.writerow([uid, vid, pdur, vdur, d, hour, round(ratio, 6), is_valid])
            valid += 1
            if total % 2_000_000 == 0:
                out.flush()
                print(f"    ... {total:,} rows, {time.time()-t0:.0f}s", flush=True)
    out.close()
    return dict(total=total, loaded=valid, dup=0, bad=bad, seconds=round(time.time() - t0, 1))


def clean_video_daily():
    src = os.path.join(RAW, "item_daily_features.csv.gz")
    dst = os.path.join(CLEAN, "video_daily.csv")
    cols = ["video_id", "date", "author_id", "video_type", "upload_type", "upload_dt",
            "visible_status", "video_duration", "video_width", "video_height",
            "music_id", "video_tag_id", "video_tag_name",
            "show_cnt", "show_user_num", "play_cnt", "play_user_num", "play_duration",
            "complete_play_cnt", "complete_play_user_num", "valid_play_cnt", "valid_play_user_num",
            "long_time_play_cnt", "long_time_play_user_num", "short_time_play_cnt",
            "short_time_play_user_num", "play_progress", "like_cnt", "like_user_num",
            "double_click_cnt", "cancel_like_cnt", "cancel_like_user_num"]
    import pandas as pd
    df = pd.read_csv(src, low_memory=False, usecols=cols)
    # 列名统一
    df = df.rename(columns={"date": "d", "video_duration": "video_duration_ms"})
    df = df[pd.to_numeric(df["video_id"], errors="coerce").notna()]
    df["video_id"] = df["video_id"].astype(int)
    df.to_csv(dst, index=False)
    return dict(rows=int(len(df)))


def load_files(conn):
    """用 LOAD DATA LOCAL INFILE 灌数据"""
    try:
        with get_conn(autocommit=True) as c2:
            c2.cursor().execute("SET GLOBAL local_infile=ON")
    except Exception as e:
        print("(local_infile 开关设置跳过:", e, ")")
    cur = conn.cursor()
    # 注意：play_log 清洗文件无表头；video_daily 清洗文件含表头需 IGNORE 1 LINES
    for table, fname, ignore in (("play_log", "play_log.csv", 0),
                                 ("video_daily", "video_daily.csv", 1)):
        path = os.path.join(CLEAN, fname)
        t0 = time.time()
        try:
            cur.execute(f"TRUNCATE TABLE {table}")
            sql = (f"LOAD DATA LOCAL INFILE '{path.replace(chr(92), '/')}' INTO TABLE {table} "
                   f"FIELDS TERMINATED BY ',' ENCLOSED BY '\"' "
                   f"LINES TERMINATED BY '\\n'"
                   + (f" IGNORE {ignore} LINES" if ignore else ""))
            cur.execute(sql)
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            cnt = cur.fetchone()[0]
            print(f"{table}: loaded {cnt:,} rows in {time.time()-t0:.0f}s")
        except Exception as e:
            print(f"{table} LOAD 失败: {e}")
            raise
    cur.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-load", action="store_true")
    args = ap.parse_args()

    dst_play = os.path.join(CLEAN, "play_log.csv")
    dst_vd = os.path.join(CLEAN, "video_daily.csv")
    if os.path.exists(dst_play) and os.path.getsize(dst_play) > 100_000_000:
        print("[1] 播放流清洗文件已存在，跳过")
    else:
        print("[1] 清洗播放流 ...")
        st = clean_play_log()
        print(f"    big_matrix 总行 {st['total']:,}，写入 {st['loaded']:,} 条有效/唯一（重复 {st['dup']:,}，坏行 {st['bad']:,}），耗时 {st['seconds']}s")
    if os.path.exists(dst_vd) and os.path.getsize(dst_vd) > 1_000_000:
        print("[2] 内容日维表清洗文件已存在，跳过")
    else:
        print("[2] 清洗内容日维表 ...")
        st2 = clean_video_daily()
        print(f"    video_daily 写入 {st2['rows']:,} 行")

    if not args.skip_load:
        print("[3] 建表 ...")
        conn = get_conn(autocommit=True)
        run_schema(conn.cursor())
        print("[4] LOAD DATA 灌库 ...")
        load_files(conn)
        conn.close()
    print("完成")


if __name__ == "__main__":
    main()
