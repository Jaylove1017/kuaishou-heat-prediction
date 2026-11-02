# -*- coding: utf-8 -*-
"""执行 sql/analysis.sql 的全部查询段并把结果归档到 outputs/base_metrics.json
用法: python src/run_analysis.py
"""
import json
import os
import re
import sys

import pymysql

sys.stdout.reconfigure(encoding="utf-8")
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DB = dict(host=os.environ.get("MYSQL_HOST", "127.0.0.1"),
          port=int(os.environ.get("MYSQL_PORT", "3307")),
          user=os.environ.get("MYSQL_USER", "root"),
          password=os.environ.get("MYSQL_PASSWORD", ""),
          database=os.environ.get("MYSQL_DB", "kuairec"))


def clean_val(v):
    from decimal import Decimal
    if v is None:
        return None
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v) if isinstance(v, float) else int(v)
    return v


def main():
    conn = pymysql.connect(host=DB["host"], port=DB["port"], user=DB["user"],
                           password=DB["password"], database=DB["database"], charset="utf8mb4")
    cur = conn.cursor()
    with open(os.path.join(PROJ, "sql", "analysis.sql"), encoding="utf-8") as f:
        text = f.read()

    blocks = []
    cur_label = None
    buf = []
    for line in text.splitlines():
        m = re.match(r"^-- Q:\s*(.*)", line)
        if m:
            if cur_label and "".join(buf).strip():
                blocks.append((cur_label, "\n".join(buf)))
            cur_label = m.group(1).strip()
            buf = []
        else:
            buf.append(line)
    if cur_label and "".join(buf).strip():
        blocks.append((cur_label, "\n".join(buf)))

    out = {}
    for label, sql in blocks:
        try:
            cur.execute(sql)
            cols = [c[0] for c in cur.description]
            rows = cur.fetchall()
        except Exception as e:
            print(f"[FAIL] {label}: {e}")
            continue
        key = re.split(r"[（(]", label)[0].strip()
        if len(rows) == 1 and len(cols) > 1:
            rec = {cols[i]: clean_val(rows[0][i]) for i in range(len(cols))}
        else:
            rec = [{cols[i]: clean_val(r[i]) for i in range(len(cols))} for r in rows]
        out[key] = rec
        print(f"[OK] {label}")

    os.makedirs(os.path.join(PROJ, "outputs"), exist_ok=True)
    with open(os.path.join(PROJ, "outputs", "base_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    conn.close()
    print("saved outputs/base_metrics.json")


if __name__ == "__main__":
    main()
