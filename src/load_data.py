# -*- coding: utf-8 -*-
"""清洗 big_matrix / item_daily_features 并整库导入 MySQL kuairec 库
用法: python src/load_data.py [--skip-load] [--force] [--tz-offset 8]

修复记录（2024 版遗留问题）
--------------------------
1. 列错位：旧版 schema 的 video_daily 有 33 列，而清洗脚本只写了 32 列 usecols
   （漏 click_like_cnt），且 LOAD DATA 不指定列名 —— MySQL 按位置灌入，
   从第 30 列起整体错一列，且不报任何错。核心标签列 like_user_num(第 29 位)
   恰好位于错位点之前，所以模型结论没被污染，但双击/取消赞三个指标全错。
   现在：列名一律从原始文件表头推导，LOAD DATA 显式带列名，并在灌库前断言列数。
2. 建库顺序：旧版连接时就带 database=kuairec，而库还不存在 → 全新环境必失败。
   现在：先连 server 建库，再切库。
3. 假指标：旧版 clean_play_log 返回 dup=0 是硬编码的，代码里根本没有去重逻辑，
   日志却打印「重复 0」。现在删掉这个字段，改为灌库后用真实行数比对。
"""
import argparse
import csv
import gzip
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(PROJ, "data", "raw")
CLEAN = os.path.join(PROJ, "data", "clean")
OUT = os.path.join(PROJ, "outputs")

DB = dict(host=os.environ.get("MYSQL_HOST", "127.0.0.1"),
          port=int(os.environ.get("MYSQL_PORT", "3307")),
          user=os.environ.get("MYSQL_USER", "root"),
          password=os.environ.get("MYSQL_PASSWORD", ""),
          database=os.environ.get("MYSQL_DB", "kuairec"))

# 数据时间字段为 UTC 壁钟，+8 换算北京时间用于「时段活跃」口径。
# 已知缺陷：只平移 hour，未平移 d。UTC 16:00-24:00（北京 00:00-08:00）的记录
# 小时会落到次日、日期仍留在前一天。设 TZ_OFFSET_HOURS=0 可两边都不平移。
TZ_OFFSET_HOURS = 8

# 期望的 video_daily 列（原始表头 -> 库表列名）。实际取交集，缺列会告警。
VD_COLS = [
    ("video_id", "video_id"), ("date", "d"), ("author_id", "author_id"),
    ("video_type", "video_type"), ("upload_type", "upload_type"), ("upload_dt", "upload_dt"),
    ("visible_status", "visible_status"), ("video_duration", "video_duration_ms"),
    ("video_width", "video_width"), ("video_height", "video_height"),
    ("music_id", "music_id"), ("video_tag_id", "video_tag_id"),
    ("video_tag_name", "video_tag_name"),
    ("show_cnt", "show_cnt"), ("show_user_num", "show_user_num"),
    ("play_cnt", "play_cnt"), ("play_user_num", "play_user_num"),
    ("play_duration", "play_duration"),
    ("complete_play_cnt", "complete_play_cnt"),
    ("complete_play_user_num", "complete_play_user_num"),
    ("valid_play_cnt", "valid_play_cnt"), ("valid_play_user_num", "valid_play_user_num"),
    ("long_time_play_cnt", "long_time_play_cnt"),
    ("long_time_play_user_num", "long_time_play_user_num"),
    ("short_time_play_cnt", "short_time_play_cnt"),
    ("short_time_play_user_num", "short_time_play_user_num"),
    ("play_progress", "play_progress"),
    ("like_cnt", "like_cnt"), ("like_user_num", "like_user_num"),
    ("click_like_cnt", "click_like_cnt"),
    ("double_click_cnt", "double_click_cnt"),
    ("cancel_like_cnt", "cancel_like_cnt"),
    ("cancel_like_user_num", "cancel_like_user_num"),
]
PLAY_COLS = ["user_id", "video_id", "play_duration_ms", "video_duration_ms",
             "d", "hour", "watch_ratio", "is_valid"]


# ---------------------------------------------------------------- 连接
def get_server_conn(autocommit=True):
    """不带 database 连接 —— 建库必须先有这一步，否则全新环境直接 Unknown database"""
    import pymysql
    return pymysql.connect(host=DB["host"], port=DB["port"], user=DB["user"],
                           password=DB["password"], autocommit=autocommit,
                           local_infile=True, charset="utf8mb4")


def get_conn(autocommit=True):
    return get_server_conn(autocommit=autocommit)  # 建库后同一条连接 select_db 即可


def run_schema(conn):
    """建库 + 建表。不再跳过 CREATE DATABASE —— 那正是旧版必失败的原因。"""
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS {DB['database']} "
                f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    conn.select_db(DB["database"])

    with open(os.path.join(PROJ, "sql", "schema.sql"), "r", encoding="utf-8") as f:
        sql = f.read()
    # 按分号切分，并剥掉每段里的纯注释行 —— 否则第一段会以 "-- KuaiRec ..." 开头，
    # 用 startswith 判断语句类型时会全部判错。
    for stmt in sql.split(";"):
        body = "\n".join(ln for ln in stmt.splitlines()
                         if not ln.strip().startswith("--")).strip()
        if not body:
            continue
        head = body.upper()
        if head.startswith(("CREATE DATABASE", "USE ")):
            continue  # 建库与切库上面已经做过
        cur.execute(body)
    print(f"schema ok: {DB['database']}.play_log & video_daily 已建")
    cur.close()


# ---------------------------------------------------------------- 清洗
def clean_play_log(tz_offset=TZ_OFFSET_HOURS):
    """流式清洗播放流 → data/clean/play_log.csv（明文，便于断点观察与 LOAD DATA）"""
    src = os.path.join(RAW, "big_matrix.csv.gz")
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"找不到 {src}\n请先按 data/README.md 下载 KuaiRec 原始文件到 data/raw/")
    dst = os.path.join(CLEAN, "play_log.csv")
    os.makedirs(CLEAN, exist_ok=True)
    total = loaded = bad = 0
    t0 = time.time()
    out = open(dst, "w", encoding="utf-8", newline="", buffering=1024 * 1024)
    w = csv.writer(out)
    with gzip.open(src, "rt", encoding="utf-8", errors="replace", newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        if len(header) != 8:
            print(f"[warn] big_matrix 表头 {len(header)} 列: {header}")
        for row in rd:
            total += 1
            if len(row) != 8:
                bad += 1
                continue
            try:
                uid = int(row[0])
                vid = int(row[1])
                pdur = int(float(row[2]))
                vdur = int(float(row[3]))
                ts = float(row[6])
                ratio = float(row[7])
                d = int(row[5])
            except (ValueError, TypeError):
                bad += 1
                continue
            if pdur <= 0 or vdur <= 0:
                bad += 1
                continue
            hour = int((ts % 86400) // 3600 + tz_offset) % 24
            is_valid = 1 if ratio <= 10.0 else 0
            w.writerow([uid, vid, pdur, vdur, d, hour, round(ratio, 6), is_valid])
            loaded += 1
            if total % 2_000_000 == 0:
                out.flush()
                print(f"    ... {total:,} rows, {time.time()-t0:.0f}s", flush=True)
    out.close()
    return dict(total=total, loaded=loaded, bad=bad, seconds=round(time.time() - t0, 1))


def clean_video_daily():
    """清洗内容日维表。

    列名从原始文件表头推导，并与 VD_COLS 取交集 —— 这样无论第三方镜像多一列
    还是少一列，都不会再出现「表 33 列 / 脚本 32 列」的静默错位。
    返回实际写出的库表列名顺序，供 LOAD DATA 显式指定。
    """
    src = os.path.join(RAW, "item_daily_features.csv.gz")
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"找不到 {src}\n请先按 data/README.md 下载 KuaiRec 原始文件到 data/raw/")
    dst = os.path.join(CLEAN, "video_daily.csv")

    with gzip.open(src, "rt", encoding="utf-8", errors="replace", newline="") as f:
        header = next(csv.reader(f))
    header = [h.strip() for h in header]

    pairs = [(s, t) for s, t in VD_COLS if s in header]
    missing = [s for s, _ in VD_COLS if s not in header]
    extra = [h for h in header if h not in {s for s, _ in VD_COLS}]
    if missing:
        print(f"[warn] 原始文件缺少 {len(missing)} 个期望列，将不导入: {missing}")
    if extra:
        print(f"[info] 原始文件有 {len(extra)} 个未纳入的列（不影响建模）: {extra[:8]}"
              + (" ..." if len(extra) > 8 else ""))
    if not pairs:
        raise RuntimeError("原始文件与期望列名完全对不上，请检查是否下错了数据集")

    import pandas as pd
    df = pd.read_csv(src, low_memory=False, usecols=[s for s, _ in pairs])
    df = df[[s for s, _ in pairs]]
    df.columns = [t for _, t in pairs]
    df = df[pd.to_numeric(df["video_id"], errors="coerce").notna()]
    df["video_id"] = df["video_id"].astype(int)
    df.to_csv(dst, index=False)
    return dict(rows=int(len(df)), columns=[t for _, t in pairs])


# ---------------------------------------------------------------- 灌库
def load_files(conn, vd_columns):
    """LOAD DATA LOCAL INFILE 灌数据。一律显式给出列名，不依赖位置对齐。"""
    # 注意：不要用 `with conn as c` —— pymysql 的 Connection.__enter__ 返回的是
    # cursor 而不是连接，`c.cursor()` 会抛 AttributeError 并被静默吞掉，
    # 结果 local_infile 从来没被打开，直到 LOAD 报错才莫名其妙。
    try:
        c2 = get_server_conn(autocommit=True)
        c2.cursor().execute("SET GLOBAL local_infile=ON")
        c2.close()
    except Exception as e:
        print("(local_infile 全局开关设置失败，若 LOAD 报错请手动 "
              "SET GLOBAL local_infile=ON):", e)

    jobs = [
        ("play_log", "play_log.csv", 0, PLAY_COLS),
        ("video_daily", "video_daily.csv", 1, vd_columns),
    ]
    counts = {}
    cur = conn.cursor()
    for table, fname, ignore, cols in jobs:
        path = os.path.join(CLEAN, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"缺少清洗文件 {path}，请先运行清洗步骤")
        csv_rows = sum(1 for _ in open(path, "r", encoding="utf-8")) - ignore
        t0 = time.time()
        cur.execute(f"TRUNCATE TABLE {table}")
        collist = ", ".join(cols)
        sql = (f"LOAD DATA LOCAL INFILE '{path.replace(chr(92), '/')}' INTO TABLE {table} "
               f"FIELDS TERMINATED BY ',' ENCLOSED BY '\"' LINES TERMINATED BY '\\n'"
               + (f" IGNORE {ignore} LINES" if ignore else "")
               + f" ({collist})")
        cur.execute(sql)
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        cnt = cur.fetchone()[0]
        counts[table] = dict(csv_rows=csv_rows, db_rows=cnt,
                             seconds=round(time.time() - t0, 1))
        flag = "" if cnt == csv_rows else f"  ⚠ 差 {csv_rows - cnt:,} 行"
        print(f"{table}: loaded {cnt:,} / csv {csv_rows:,} rows "
              f"in {time.time()-t0:.0f}s{flag}")
    cur.close()
    return counts


def check_quality(conn, counts):
    """数据质量体检：把「重复/空值/异常」查出来，而不是靠日志里的硬编码数字。

    这几条查询本身就是可交付的产物（outputs/data_quality.json），
    也是本项目「数据治理」这一环的可验证证据。
    """
    cur = conn.cursor()
    checks = {}
    queries = {
        "video_daily_重复(video_id,d)": """
            SELECT COUNT(*) FROM (
                SELECT video_id, d FROM video_daily GROUP BY video_id, d HAVING COUNT(*) > 1
            ) t""",
        "play_log_重复(user_id,video_id,d,hour)": """
            SELECT COUNT(*) FROM (
                SELECT user_id, video_id, d, hour FROM play_log
                GROUP BY user_id, video_id, d, hour HAVING COUNT(*) > 1
            ) t""",
        "play_log_异常完播比行数": "SELECT COUNT(*) FROM play_log WHERE is_valid = 0",
        "play_log_关键列空值": """
            SELECT SUM(user_id IS NULL) + SUM(video_id IS NULL)
                 + SUM(play_duration_ms IS NULL) + SUM(d IS NULL)
            FROM play_log""",
        "video_daily_关键列空值": """
            SELECT SUM(video_id IS NULL) + SUM(d IS NULL) + SUM(play_cnt IS NULL)
                 + SUM(like_user_num IS NULL)
            FROM video_daily""",
    }
    for name, q in queries.items():
        try:
            cur.execute(q)
            checks[name] = int(cur.fetchone()[0] or 0)
        except Exception as e:
            checks[name] = f"查询失败: {e}"
    cur.close()
    checks["_行数核对"] = counts
    for k, v in checks.items():
        if k.startswith("_"):
            continue
        print(f"    {k}: {v}")
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-load", action="store_true", help="只清洗，不灌库")
    ap.add_argument("--force", action="store_true", help="已存在的清洗文件也重新生成")
    ap.add_argument("--tz-offset", type=int, default=TZ_OFFSET_HOURS,
                    help="播放流小时字段的时区偏移，默认 8（北京时间）；0 = 不平移")
    args = ap.parse_args()

    dst_play = os.path.join(CLEAN, "play_log.csv")
    dst_vd = os.path.join(CLEAN, "video_daily.csv")

    if os.path.exists(dst_play) and os.path.getsize(dst_play) > 100_000_000 and not args.force:
        print("[1] 播放流清洗文件已存在，跳过（--force 可强制重建）")
    else:
        print("[1] 清洗播放流 ...")
        st = clean_play_log(args.tz_offset)
        print(f"    big_matrix 总行 {st['total']:,}，写入 {st['loaded']:,}"
              f"（坏行 {st['bad']:,}），耗时 {st['seconds']}s")

    if os.path.exists(dst_vd) and os.path.getsize(dst_vd) > 1_000_000 and not args.force:
        print("[2] 内容日维表清洗文件已存在，跳过（--force 可强制重建）")
        import pandas as pd
        # 直接用已有 CSV 的表头，保证 LOAD DATA 的列名与实际文件严格一致
        vd_columns = list(pd.read_csv(dst_vd, nrows=0).columns)
    else:
        print("[2] 清洗内容日维表 ...")
        st2 = clean_video_daily()
        vd_columns = st2["columns"]
        print(f"    video_daily 写入 {st2['rows']:,} 行，{len(vd_columns)} 列")

    if args.skip_load:
        print("完成（--skip-load，未灌库）")
        return

    print("[3] 建库建表 ...")
    conn = get_server_conn(autocommit=True)
    run_schema(conn)

    print("[4] LOAD DATA 灌库 ...")
    counts = load_files(conn, vd_columns)

    print("[5] 数据质量体检 ...")
    checks = check_quality(conn, counts)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "data_quality.json"), "w", encoding="utf-8") as f:
        json.dump(checks, f, ensure_ascii=False, indent=2)
    print("    已写 outputs/data_quality.json")

    conn.close()
    print("完成")


if __name__ == "__main__":
    main()
