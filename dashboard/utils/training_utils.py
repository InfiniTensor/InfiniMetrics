"""Training page utilities."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def load_training_runs(data_loader):
    """Load all training-related test runs"""
    runs = data_loader.list_test_runs("train")

    if not runs:
        all_runs = data_loader.list_test_runs()
        runs = [
            r
            for r in all_runs
            if any(
                keyword in str(r.get("path", "")).lower()
                or keyword in r.get("testcase", "").lower()
                for keyword in [
                    "/train/",
                    "/training/",
                    "train.",
                    "megatron",
                    "lora",
                    "sft",
                ]
            )
        ]
    return runs


def filter_runs(runs, selected_fw, selected_models, selected_dev, only_success):
    """Apply filters to runs"""
    return [
        r
        for r in runs
        if (
            not selected_fw
            or r.get("config", {}).get("framework", "unknown") in selected_fw
        )
        and (
            not selected_models
            or r.get("config", {}).get("model", "unknown") in selected_models
        )
        and (not selected_dev or r.get("device_used", 1) in selected_dev)
        and (not only_success or r.get("success", False))
    ]


def create_run_options(runs):
    """Create run selection options"""
    return {
        f"{r.get('config', {}).get('framework', 'unknown')}/"
        f"{r.get('config', {}).get('model', 'unknown')} | "
        f"{r.get('device_used', '?')}GPU | "
        f"{r.get('time', '')[:16]}": i
        for i, r in enumerate(runs)
    }


def load_selected_runs(data_loader, filtered_runs, options, selected_labels):
    """Load the selected test run"""
    selected_runs = []
    for label in selected_labels:
        idx = options[label]
        run_info = filtered_runs[idx].copy()
        run_info["data"] = data_loader.load_test_result(run_info["path"])
        selected_runs.append(run_info)
    return selected_runs


def get_metric_dataframe(run, metric_key):
    """Get metric dataframe"""
    metrics = run["data"].get("metrics", [])
    return next(
        (
            m
            for m in metrics
            if metric_key in m.get("name", "") and m.get("data") is not None
        ),
        None,
    )


def apply_smoothing(df, smoothing):
    """Apply smoothing to dataframe"""
    if smoothing > 1 and len(df) > smoothing:
        df = df.copy()
        df.iloc[:, 1] = df.iloc[:, 1].rolling(window=smoothing, min_periods=1).mean()
    return df


def create_training_summary(test_result: dict) -> pd.DataFrame:
    """Create configuration summary for training tests"""
    rows = []

    # Environment info
    try:
        env = test_result.get("environment", {})
        if "cluster" in env and len(env["cluster"]) > 0:
            acc = env["cluster"][0]["machine"]["accelerators"][0]
            rows.extend(
                [
                    {"指标": "加速卡", "数值": str(acc.get("model", "Unknown"))},
                    {"指标": "卡数", "数值": str(acc.get("count", "Unknown"))},
                    {"指标": "显存/卡", "数值": f"{acc.get('memory_gb_per_card','?')} GB"},
                ]
            )
    except Exception as e:
        st.warning(f"解析环境信息失败: {e}")

    # Config info
    cfg = test_result.get("config", {})
    train_args = cfg.get("train_args", {})

    rows.extend(
        [
            {"指标": "框架", "数值": str(cfg.get("framework", "unknown"))},
            {"指标": "模型", "数值": str(cfg.get("model", "unknown"))},
        ]
    )

    # Parallel config
    parallel = train_args.get("parallel", {})
    rows.append(
        {
            "指标": "并行配置",
            "数值": f"DP={parallel.get('dp', 1)}, TP={parallel.get('tp', 1)}, PP={parallel.get('pp', 1)}",
        }
    )

    # Other configs
    config_items = [
        ("MBS/GBS", f"{train_args.get('mbs', '?')}/{train_args.get('gbs', '?')}"),
        ("序列长度", str(train_args.get("seq_len", "?"))),
        ("隐藏层大小", str(train_args.get("hidden_size", "?"))),
        ("层数", str(train_args.get("num_layers", "?"))),
        ("精度", str(train_args.get("precision", "?"))),
        ("预热迭代", str(cfg.get("warmup_iterations", "?"))),
        ("训练迭代", str(train_args.get("train_iters", "?"))),
    ]

    for label, value in config_items:
        rows.append({"指标": label, "数值": value})

    # Scalar metrics
    for m in test_result.get("metrics", []):
        if m.get("type") == "scalar":
            value = m.get("value", "")
            unit = m.get("unit", "")
            rows.append(
                {
                    "指标": str(m.get("name")),
                    "数值": f"{value} {unit}" if unit and value != "" else str(value),
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["数值"] = df["数值"].astype(str)
    return df
