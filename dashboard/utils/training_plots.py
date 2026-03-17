"""Training plot functions."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils.training_utils import get_metric_dataframe, apply_smoothing
from utils.visualizations import create_gauge_chart


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
    target_metric = get_metric_dataframe(run, metric_key)

    if not target_metric:
        st.info(f"无{title}数据")
        return

    df = target_metric["data"].copy()
    if df.empty or len(df.columns) < 2:
        st.info("数据为空")
        return

    df = apply_smoothing(df, smoothing)

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
    else:
        fig.update_yaxes(rangemode="tozero")

    st.plotly_chart(fig, use_container_width=True)


def plot_multi_metric_comparison(
    runs, metric_key, title, ylabel, unit, smoothing, y_log
):
    """Comparison of multiple operation indexes"""
    colors = px.colors.qualitative.Set2
    fig = go.Figure()
    found_data = False

    for i, run in enumerate(runs):
        target_metric = get_metric_dataframe(run, metric_key)
        if not target_metric:
            continue

        df = target_metric["data"].copy()
        if df.empty or len(df.columns) < 2:
            continue

        found_data = True
        df = apply_smoothing(df, smoothing)

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
    else:
        fig.update_yaxes(rangemode="tozero")

    st.plotly_chart(fig, use_container_width=True)


def render_memory_usage(run):
    """Render memory usage gauge"""
    memory_metric = get_metric_dataframe(run, "train.peak_memory_usage")

    if not memory_metric or not memory_metric.get("value"):
        return

    st.markdown("#### 💾 显存使用")
    value = memory_metric["value"]
    unit = memory_metric.get("unit", "GB")

    # Try to get max memory from environment
    max_value = value * 1.5
    try:
        env = run["data"].get("environment", {})
        if env and "cluster" in env and len(env["cluster"]) > 0:
            acc = env["cluster"][0].get("machine", {}).get("accelerators", [])
            if acc and len(acc) > 0:
                memory_per_card = float(acc[0].get("memory_gb_per_card", 80))
                card_count = int(acc[0].get("count", 1))
                max_value = memory_per_card * card_count
            else:
                st.warning("未找到加速卡信息，使用默认显存上限")
        else:
            st.warning("未找到环境信息，使用默认显存上限")
    except Exception as e:
        st.warning(f"解析环境信息失败: {e}，使用默认显存上限")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        fig = create_gauge_chart(value, max_value, "峰值显存使用", "green", unit)
        st.plotly_chart(fig, use_container_width=True)


def render_throughput_comparison(selected_runs):
    """Render throughput comparison"""
    st.markdown("### 吞吐量对比")

    throughput_data = []
    for run in selected_runs:
        target_metric = get_metric_dataframe(run, "throughput")
        if not target_metric:
            continue

        df = target_metric["data"]
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


def render_config_details(selected_runs, summary_func):
    """Render configuration details"""
    for run in selected_runs:
        with st.expander(f"{run.get('run_id', 'Unknown')} - 配置与环境"):
            summary_df = summary_func(run["data"])
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
