#!/usr/bin/env python3
"""Training tests analysis page."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from common import init_page
from components.header import render_header
from utils.visualizations import create_gauge_chart

init_page("训练测试分析 | InfiniMetrics", "🏋️")


def main():
    render_header()
    st.markdown("## 🏋️ 训练性能测试分析")

    dl = st.session_state.data_loader

    # Load all training-related tests
    runs = load_training_runs(dl)

    if not runs:
        st.info("未找到训练测试结果\n请将训练测试结果放在 test_output/train/ 或 test_output/training/ 目录下")
        return

    # ---------- Sidebar Filters ----------
    with st.sidebar:
        st.markdown("### 🔍 筛选条件")

        frameworks = sorted(
            {r.get("config", {}).get("framework", "unknown") for r in runs}
        )
        models = sorted({r.get("config", {}).get("model", "unknown") for r in runs})
        device_counts = sorted({r.get("device_used", 1) for r in runs})

        selected_fw = st.multiselect("框架", frameworks, default=frameworks)
        selected_models = st.multiselect("模型", models, default=models)
        selected_dev = st.multiselect("设备数", device_counts, default=device_counts)
        only_success = st.checkbox("仅显示成功测试", value=True)

        st.markdown("---")
        st.markdown("### 📈 图表选项")
        y_log = st.checkbox("Y轴对数刻度", value=False)
        smoothing = st.slider("平滑窗口", 1, 50, 5, help="对曲线进行移动平均平滑")

    # Apply filters
    filtered = filter_runs(
        runs, selected_fw, selected_models, selected_dev, only_success
    )
    st.caption(f"找到 {len(filtered)} 个训练测试")

    if not filtered:
        st.warning("没有符合条件的测试结果")
        return

    # ---------- Run Selection ----------
    options = create_run_options(filtered)
    selected = st.multiselect(
        "选择要分析的测试运行（可多选对比）",
        list(options.keys()),
        default=list(options.keys())[: min(3, len(options))],
    )

    if not selected:
        return

    # Load selected runs
    selected_runs = load_selected_runs(dl, filtered, options, selected)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 性能曲线", "📊 吞吐量对比", "📋 数据表格", "🔍 详细配置"])

    with tab1:
        render_performance_curves(selected_runs, smoothing, y_log)

    with tab2:
        render_throughput_comparison(selected_runs)

    with tab3:
        render_data_tables(selected_runs)

    with tab4:
        render_config_details(selected_runs)


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
    options = {}
    for i, r in enumerate(runs):
        config = r.get("config", {})
        framework = config.get("framework", "unknown")
        model = config.get("model", "unknown")
        device = r.get("device_used", "?")
        time_str = r.get("time", "")[5:16] if r.get("time") else "unknown"
        run_id = r.get("run_id", "")[:8]

        label = f"{framework}/{model} | {device}GPU | {time_str} | {run_id}"
        options[label] = i

    return options


def load_selected_runs(data_loader, filtered_runs, options, selected_labels):
    """Load the selected test run"""
    selected_runs = []
    for label in selected_labels:
        idx = options[label]
        run_info = filtered_runs[idx].copy()
        run_info["data"] = data_loader.load_test_result(run_info["path"])
        selected_runs.append(run_info)

    return selected_runs


def render_performance_curves(selected_runs, smoothing, y_log):
    """Render performance curves"""
    st.markdown("### 训练指标曲线")

    metrics = [
        ("train.loss", "Loss", "损失值", ""),
        ("train.ppl", "Perplexity", "困惑度", ""),
        ("train.throughput", "Throughput", "吞吐量", "tokens/s/GPU"),
    ]

    cols = st.columns(3)

    for idx, (metric_key, title, ylabel, unit) in enumerate(metrics):
        with cols[idx]:
            st.markdown(f"**{title}**")

            if len(selected_runs) == 1:
                plot_single_metric(
                    selected_runs[0], metric_key, title, ylabel, unit, smoothing, y_log
                )
            else:
                plot_multi_metric_comparison(
                    selected_runs, metric_key, title, ylabel, unit, smoothing, y_log
                )

    # Memory usage (only for single run)
    if len(selected_runs) == 1:
        render_memory_usage(selected_runs[0])


def plot_single_metric(run, metric_key, title, ylabel, unit, smoothing, y_log):
    """Draw a curve for a single indicator"""
    metrics = run["data"].get("metrics", [])
    target_metric = next(
        (
            m
            for m in metrics
            if metric_key in m.get("name", "") and m.get("data") is not None
        ),
        None,
    )

    if not target_metric:
        st.info(f"无{title}数据")
        return

    df = target_metric["data"].copy()
    if df.empty or len(df.columns) < 2:
        st.info("数据为空")
        return

    # Apply smoothing
    if smoothing > 1 and len(df) > smoothing:
        df.iloc[:, 1] = df.iloc[:, 1].rolling(window=smoothing, min_periods=1).mean()

    # Create a chart
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[df.columns[0]],
            y=df[df.columns[1]],
            mode="lines",
            name=title,
            line=dict(width=2),
        )
    )

    fig.update_layout(
        title=f"{title} - {run.get('config', {}).get('framework', '')}",
        xaxis_title="Iteration",
        yaxis_title=f"{ylabel} {unit}",
        template="plotly_white",
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
    )

    if y_log:
        fig.update_yaxes(type="log")

    st.plotly_chart(fig, use_container_width=True)


def plot_multi_metric_comparison(
    runs, metric_key, title, ylabel, unit, smoothing, y_log
):
    """Comparison of multiple operation indexes"""
    colors = px.colors.qualitative.Set2

    fig = go.Figure()
    found_data = False

    for i, run in enumerate(runs):
        metrics = run["data"].get("metrics", [])
        target_metric = next(
            (
                m
                for m in metrics
                if metric_key in m.get("name", "") and m.get("data") is not None
            ),
            None,
        )

        if not target_metric:
            continue

        df = target_metric["data"].copy()
        if df.empty or len(df.columns) < 2:
            continue

        found_data = True

        # Apply smoothing
        if smoothing > 1 and len(df) > smoothing:
            df.iloc[:, 1] = (
                df.iloc[:, 1].rolling(window=smoothing, min_periods=1).mean()
            )

        # Building legend labels
        config = run.get("config", {})
        framework = config.get("framework", "unknown")
        model = config.get("model", "unknown")
        device = run.get("device_used", "?")

        fig.add_trace(
            go.Scatter(
                x=df[df.columns[0]],
                y=df[df.columns[1]],
                mode="lines",
                name=f"{framework}/{model} ({device}GPU)",
                line=dict(color=colors[i % len(colors)], width=2),
            )
        )

    if not found_data:
        st.info(f"无{title}数据")
        return

    fig.update_layout(
        title=f"{title} 对比",
        xaxis_title="Iteration",
        yaxis_title=f"{ylabel} {unit}",
        template="plotly_white",
        height=350,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=40, r=20, t=40, b=40),
    )

    if y_log:
        fig.update_yaxes(type="log")

    st.plotly_chart(fig, use_container_width=True)


def render_memory_usage(run):
    """Render memory usage gauge"""
    memory_metric = next(
        (
            m
            for m in run["data"].get("metrics", [])
            if m.get("name") == "train.peak_memory_usage"
        ),
        None,
    )

    if not memory_metric or not memory_metric.get("value"):
        return

    st.markdown("#### 💾 显存使用")
    value = memory_metric["value"]
    unit = memory_metric.get("unit", "GB")

    # Try to get max memory from environment
    max_value = value * 1.5
    try:
        env = run["data"].get("environment", {})
        acc = env["cluster"][0]["machine"]["accelerators"][0]
        max_value = float(acc.get("memory_gb_per_card", 80)) * int(acc.get("count", 1))
    except:
        pass

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        fig = create_gauge_chart(value, max_value, "峰值显存使用", "green", unit)
        st.plotly_chart(fig, use_container_width=True)


def render_throughput_comparison(selected_runs):
    """Render throughput comparison"""
    st.markdown("### 吞吐量对比")

    throughput_data = []
    for run in selected_runs:
        metrics = run["data"].get("metrics", [])
        throughput_metric = next(
            (
                m
                for m in metrics
                if "throughput" in m.get("name", "").lower()
                and m.get("data") is not None
            ),
            None,
        )

        if not throughput_metric:
            continue

        df = throughput_metric["data"]
        if df.empty or len(df.columns) < 2:
            continue

        avg_tput = df[df.columns[1]].mean()
        peak_tput = df[df.columns[1]].max()

        config = run.get("config", {})
        throughput_data.append(
            {
                "运行": f"{config.get('framework', 'unknown')}/{config.get('model', 'unknown')} ({run.get('device_used', '?')}GPU)",
                "平均吞吐量 (tokens/s/GPU)": round(avg_tput, 2),
                "峰值吞吐量 (tokens/s/GPU)": round(peak_tput, 2),
            }
        )

    if not throughput_data:
        st.info("无可对比的吞吐量数据")
        return

    # Display table
    df = pd.DataFrame(throughput_data)
    st.dataframe(df, width="stretch", hide_index=True)

    # Bar chart
    fig = go.Figure()
    for i, data in enumerate(throughput_data):
        fig.add_trace(
            go.Bar(
                name=data["运行"],
                x=["平均吞吐量", "峰值吞吐量"],
                y=[data["平均吞吐量 (tokens/s/GPU)"], data["峰值吞吐量 (tokens/s/GPU)"]],
                text=[data["平均吞吐量 (tokens/s/GPU)"], data["峰值吞吐量 (tokens/s/GPU)"]],
                textposition="auto",
            )
        )

    fig.update_layout(
        title="吞吐量对比",
        barmode="group",
        template="plotly_white",
        height=400,
        yaxis_title="tokens/s/GPU",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_data_tables(selected_runs):
    """Render data tables"""
    for run in selected_runs:
        with st.expander(f"{run.get('run_id', 'Unknown')} - 原始数据"):
            for metric in run["data"].get("metrics", []):
                if metric.get("data") is not None:
                    df = metric["data"].copy()
                    st.markdown(f"**{metric.get('name', 'Unknown')}**")

                    if len(df.columns) == 2:
                        df.columns = ["Iteration", metric.get("name", "Value")]

                    st.dataframe(df, width="stretch", hide_index=True)


def render_config_details(selected_runs):
    """Render configuration details"""
    for run in selected_runs:
        with st.expander(f"{run.get('run_id', 'Unknown')} - 配置与环境"):
            summary_df = create_training_summary(run["data"])
            if not summary_df.empty:
                st.markdown("**配置摘要**")
                st.dataframe(summary_df, width="stretch", hide_index=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**完整配置**")
                st.json(run["data"].get("config", {}))
            with col2:
                st.markdown("**环境信息**")
                st.json(run["data"].get("environment", {}))

            if run["data"].get("resolved"):
                st.markdown("**解析信息**")
                st.json(run["data"].get("resolved", {}))


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
    except:
        pass

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


if __name__ == "__main__":
    main()
