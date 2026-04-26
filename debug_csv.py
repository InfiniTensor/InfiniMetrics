#!/usr/bin/env python3
"""
诊断脚本2：直接检查各维度 CSV 文件内容和列名。
cd ~/InfiniBench && python debug_csv.py
"""
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("./output")
SEP = "─" * 70

DIRS = {
    "comm": OUTPUT_DIR / "comm",
    "hardware": OUTPUT_DIR / "hardware",
    "operator": OUTPUT_DIR / "operator",
    "training": OUTPUT_DIR / "training",
    "infer": OUTPUT_DIR / "infer",
    "output_root": OUTPUT_DIR,  # output/ 根目录下也有 csv
}

for dim, d in DIRS.items():
    csvs = list(d.glob("*.csv")) if d.exists() else []
    if not csvs:
        continue
    print(f"\n{SEP}")
    print(f"[{dim}] {d}")
    print(SEP)
    for csv in csvs[:4]:
        print(f"\n  >> {csv.name}")
        try:
            df = pd.read_csv(csv)
            print(f"     columns : {list(df.columns)}")
            print(f"     shape   : {df.shape}")
            print(df.head(3).to_string(index=False))
        except Exception as e:
            print(f"     ERROR: {e}")
