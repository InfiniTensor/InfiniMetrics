#!/usr/bin/env python3
"""Summary table functions for different test types."""

import pandas as pd
from typing import Dict, Any


def create_summary_table_comm(test_result: Dict[str, Any]) -> pd.DataFrame:
    """Create summary table for communication tests."""
    summary_data = []

    # Hardware summary
    if "environment" in test_result:
        env = test_result["environment"]
        if "cluster" in env and len(env["cluster"]) > 0:
            machine = env["cluster"][0]["machine"]
            accelerators = machine.get("accelerators", [])
            if accelerators:
                acc = accelerators[0]
                summary_data.append(
                    {"指标": "GPU型号", "数值": str(acc.get("model", "Unknown"))}
                )
                summary_data.append(
                    {"指标": "GPU数量", "数值": str(acc.get("count", "Unknown"))}
                )
                summary_data.append(
                    {
                        "指标": "显存/卡",
                        "数值": f"{acc.get('memory_gb_per_card', 'Unknown')} GB",
                    }
                )
                summary_data.append(
                    {"指标": "CUDA版本", "数值": str(acc.get("cuda", "Unknown"))}
                )

    # Test config summary
    config = test_result.get("config", {})
    resolved = test_result.get("resolved", {})

    device_used = (
        resolved.get("device_used")
        or config.get("device_used")
        or config.get("device_involved", "Unknown")
    )
    nodes = resolved.get("nodes") or config.get("nodes", 1)

    summary_data.append({"指标": "算子", "数值": str(config.get("operator", "Unknown"))})
    summary_data.append({"指标": "设备数", "数值": str(device_used)})
    summary_data.append({"指标": "节点数", "数值": str(nodes)})
    summary_data.append(
        {"指标": "预热迭代", "数值": str(config.get("warmup_iterations", "Unknown"))}
    )
    summary_data.append(
        {"指标": "测量迭代", "数值": str(config.get("measured_iterations", "Unknown"))}
    )

    # Performance summary
    for metric in test_result.get("metrics", []):
        if metric.get("name") == "comm.bandwidth" and metric.get("data") is not None:
            df = metric["data"]
            if "bandwidth_gbs" in df.columns:
                avg_bw = df["bandwidth_gbs"].mean()
                max_bw = df["bandwidth_gbs"].max()
                summary_data.append({"指标": "平均带宽", "数值": f"{avg_bw:.2f} GB/s"})
                summary_data.append({"指标": "峰值带宽", "数值": f"{max_bw:.2f} GB/s"})

        if metric.get("name") == "comm.latency" and metric.get("data") is not None:
            df = metric["data"]
            if "latency_us" in df.columns:
                avg_lat = df["latency_us"].mean()
                min_lat = df["latency_us"].min()
                summary_data.append({"指标": "平均延迟", "数值": f"{avg_lat:.2f} µs"})
                summary_data.append({"指标": "最小延迟", "数值": f"{min_lat:.2f} µs"})

    duration = next(
        (
            m["value"]
            for m in test_result.get("metrics", [])
            if m.get("name") == "comm.duration"
        ),
        None,
    )
    if duration:
        summary_data.append({"指标": "测试耗时", "数值": f"{duration:.2f} ms"})

    return pd.DataFrame(summary_data)


def create_summary_table_infer(test_result: dict) -> pd.DataFrame:
    """Create summary table for inference tests."""
    rows = []

    env = test_result.get("environment", {})
    try:
        acc = env["cluster"][0]["machine"]["accelerators"][0]
        rows += [
            {"指标": "加速卡", "数值": str(acc.get("model", "Unknown"))},
            {"指标": "卡数", "数值": str(acc.get("count", "Unknown"))},
            {"指标": "显存/卡", "数值": f"{acc.get('memory_gb_per_card','?')} GB"},
            {"指标": "CUDA", "数值": str(acc.get("cuda", "Unknown"))},
            {"指标": "平台", "数值": str(acc.get("type", "nvidia"))},
        ]
    except Exception:
        pass

    cfg = test_result.get("config", {})
    rows += [
        {"指标": "框架", "数值": str(cfg.get("framework", "unknown"))},
        {"指标": "模型", "数值": str(cfg.get("model", ""))},
        {
            "指标": "batch",
            "数值": str(
                (cfg.get("infer_args", {}) or {}).get("static_batch_size", "unknown")
            ),
        },
        {
            "指标": "prompt_tok",
            "数值": str(
                (cfg.get("infer_args", {}) or {}).get("prompt_token_num", "unknown")
            ),
        },
        {
            "指标": "output_tok",
            "数值": str(
                (cfg.get("infer_args", {}) or {}).get("output_token_num", "unknown")
            ),
        },
        {"指标": "warmup", "数值": str(cfg.get("warmup_iterations", "unknown"))},
        {"指标": "measured", "数值": str(cfg.get("measured_iterations", "unknown"))},
    ]

    for m in test_result.get("metrics", []):
        if m.get("type") == "scalar":
            rows.append(
                {
                    "指标": str(m.get("name", "")),
                    "数值": f"{m.get('value', '')} {m.get('unit', '')}".strip(),
                }
            )

    return pd.DataFrame(rows)


def create_summary_table_ops(test_result: dict) -> pd.DataFrame:
    """Create summary table for operator tests."""
    rows = []
    cfg = test_result.get("config", {})

    rows.append({"指标": "testcase", "数值": str(test_result.get("testcase", ""))})
    rows.append(
        {"指标": "算子", "数值": str(cfg.get("operator", cfg.get("op_name", "Unknown")))}
    )

    env = test_result.get("environment", {})
    try:
        acc = env["cluster"][0]["machine"]["accelerators"][0]
        rows += [
            {"指标": "加速卡", "数值": str(acc.get("model", "Unknown"))},
            {"指标": "卡数", "数值": str(acc.get("count", "Unknown"))},
        ]
    except Exception:
        pass

    scalars = [m for m in test_result.get("metrics", []) if m.get("type") == "scalar"]
    for m in scalars:
        rows.append(
            {
                "指标": str(m.get("name", "")),
                "数值": f"{m.get('value', '')} {m.get('unit', '')}".strip(),
            }
        )

    for k in [
        "dtype",
        "shape",
        "batch_size",
        "warmup_iterations",
        "measured_iterations",
    ]:
        if k in cfg:
            rows.append({"指标": k, "数值": str(cfg.get(k))})

    return pd.DataFrame(rows)
