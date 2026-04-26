#!/usr/bin/env python3
"""
诊断脚本：找出通信/算子/硬件数据为何无法显示。
在 InfiniBench 根目录下运行：
    python debug_dashboard.py
"""
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.append(str(project_root))

OUTPUT_DIR = Path("./output")

# ── 1. 直接扫描各维度目录下的 JSON，看关键字段 ────────────────────────────────
DIM_DIRS = {
    "comm": OUTPUT_DIR / "comm",
    "hardware": OUTPUT_DIR / "hardware",
    "operator": OUTPUT_DIR / "operator",
    "infer": OUTPUT_DIR / "infer",
    "training": OUTPUT_DIR / "training",
}

SEP = "─" * 70


def inspect_json(path: Path):
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        print(f"  [ERROR] 读取失败: {e}")
        return

    print(f"  testcase        : {data.get('testcase', '❌ 缺失')}")
    print(f"  run_id          : {data.get('run_id',   '❌ 缺失')}")
    print(f"  accelerator_types: {data.get('accelerator_types', '❌ 缺失')}")
    print(f"  success         : {data.get('success',  '❌ 缺失')}")

    metrics = data.get("metrics", [])
    print(f"  metrics 数量    : {len(metrics)}")
    for i, m in enumerate(metrics[:5]):
        name = m.get("name", "?")
        has_data = m.get("data") is not None
        print(f"    [{i}] name={name!r}  has_data={has_data}")
        if has_data and isinstance(m["data"], dict):
            cols = list(m["data"].keys()) if isinstance(m["data"], dict) else "非dict"
            print(f"         data keys={cols}")


print(SEP)
print("【1】各维度目录 JSON 文件关键字段")
print(SEP)

for dim, dirpath in DIM_DIRS.items():
    if not dirpath.exists():
        print(f"\n[{dim}] 目录不存在: {dirpath}")
        continue
    jsons = list(dirpath.glob("*.json")) + list(OUTPUT_DIR.glob(f"{dim}*.json"))
    jsons = list({p.resolve() for p in jsons})  # 去重
    print(f"\n[{dim}] 找到 {len(jsons)} 个 JSON:")
    for jp in jsons[:3]:
        print(f"  >> {jp.name}")
        inspect_json(jp)

# ── 2. 用 data_loader 实际 list_test_runs，看返回结构 ─────────────────────────
print(f"\n{SEP}")
print("【2】data_loader.list_test_runs() 返回的每条记录关键字段")
print(SEP)

try:
    from utils.data_loader import InfiniBenchDataLoader

    loader = InfiniBenchDataLoader()
    runs = loader.list_test_runs()
    print(f"总共 {len(runs)} 条 run\n")

    for r in runs:
        rid = r.get("run_id", "?")
        tc = r.get("testcase", "?")
        accs = r.get("accelerator_types", [])
        path = r.get("path", "")
        print(f"  run_id={rid}")
        print(
            f"    testcase={tc!r}  accelerator_types={accs}  path={str(path)[-60:]!r}"
        )

except Exception as e:
    print(f"[ERROR] data_loader 加载失败: {e}")
    import traceback

    traceback.print_exc()

# ── 3. 对 comm/hardware/operator 各取一个 run，实际 load_test_result ──────────
print(f"\n{SEP}")
print("【3】load_test_result 实际返回的 metrics 结构（comm/hardware/operator）")
print(SEP)

TARGET_PREFIXES = ["comm", "hardware", "operator"]

try:
    from utils.data_loader import InfiniBenchDataLoader

    loader = InfiniBenchDataLoader()
    runs = loader.list_test_runs()

    for prefix in TARGET_PREFIXES:
        matched = [
            r
            for r in runs
            if (r.get("testcase", "") or "").lower().startswith(prefix)
            or prefix in (r.get("run_id", "") or "").lower()
            or prefix in str(r.get("path", "")).lower()
        ]
        print(f"\n-- {prefix}: 匹配到 {len(matched)} 条 run")
        if not matched:
            continue
        r = matched[0]
        ident = r.get("path") or r.get("run_id")
        print(f"   identifier={str(ident)[-80:]!r}")
        try:
            data = loader.load_test_result(ident)
            metrics = data.get("metrics", [])
            print(f"   metrics 数量: {len(metrics)}")
            for i, m in enumerate(metrics):
                name = m.get("name", "?")
                has_data = m.get("data") is not None
                df = m.get("data")
                cols = (
                    list(df.columns)
                    if hasattr(df, "columns")
                    else (
                        list(df.keys()) if isinstance(df, dict) else type(df).__name__
                    )
                )
                print(f"   [{i}] name={name!r}  has_data={has_data}  columns={cols}")
        except Exception as e2:
            print(f"   [ERROR] load_test_result 失败: {e2}")
            import traceback

            traceback.print_exc()

except Exception as e:
    print(f"[ERROR]: {e}")
    import traceback

    traceback.print_exc()

print(f"\n{SEP}")
print("诊断完成")
print(SEP)
