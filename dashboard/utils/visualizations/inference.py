#!/usr/bin/env python3
"""Inference-specific visualization functions."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from .base import plot_timeseries_auto


def render_inference_metrics(selected_runs, y_log):
    """Render inference metrics charts (Latency / TTFT / Throughput)."""
    st.markdown("### 指标曲线（Latency / TTFT / Throughput）")

    # layout: 3 columns
    c1, c2, c3 = st.columns(3)

    def _plot_metric(metric_name_contains: str, container, title: str = None):
        with container:
            if len(selected_runs) == 1:
                # Single run
                run = selected_runs[0]
                metrics = run["data"].get("metrics", [])
                hit = next(
                    (
                        m
                        for m in metrics
                        if metric_name_contains in (m.get("name", ""))
                        and m.get("data") is not None
                    ),
                    None,
                )
                if hit:
                    df = hit["data"]
                    fig = plot_timeseries_auto(
                        df,
                        title=f"{hit['name']} - {run.get('config',{}).get('framework','')}",
                        y_log_scale=y_log,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"未找到 {title or metric_name_contains} 对应的 CSV")
            else:
                # Multi-run comparison: overlay lines
                st.markdown(f"**对比：{title or metric_name_contains}**")
                lines = []
                for run in selected_runs:
                    hit = next(
                        (
                            m
                            for m in run["data"].get("metrics", [])
                            if metric_name_contains in (m.get("name", ""))
                            and m.get("data") is not None
                        ),
                        None,
                    )
                    if not hit:
                        continue
                    lines.append((run, hit))

                if not lines:
                    st.info("选中的运行中没有可用数据")
                    return

                fig = go.Figure()
                colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

                for i, (run, hit) in enumerate(lines):
                    df = hit["data"]
                    xcol = df.columns[0]
                    ycol = df.columns[1] if len(df.columns) > 1 else None
                    if ycol is None:
                        continue

                    # Get mode from testcase
                    tc = run.get("testcase", "")
                    mode = (
                        "service"
                        if ("Service" in tc or "service" in tc.lower())
                        else "direct"
                    )

                    label = f"{run.get('config',{}).get('framework','unknown')}|{mode}|{run.get('device_used','?')}GPU"
                    fig.add_trace(
                        go.Scatter(
                            x=df[xcol],
                            y=df[ycol],
                            mode="lines+markers",
                            name=label,
                            line=dict(color=colors[i % len(colors)], width=2),
                            marker=dict(size=6),
                        )
                    )

                fig.update_layout(
                    title=f"{title or metric_name_contains} 对比",
                    xaxis_title="step",
                    yaxis_title=title or metric_name_contains,
                    template="plotly_white",
                    height=420,
                    hovermode="x unified",
                )
                if y_log:
                    fig.update_yaxes(type="log")
                st.plotly_chart(fig, use_container_width=True)

    _plot_metric("infer.compute_latency", c1, "Latency")
    _plot_metric("infer.ttft", c2, "TTFT")
    _plot_metric("infer.direct_throughput_tps", c3, "Throughput")


def render_memory_gauge(run):
    """Render memory usage gauge for inference."""
    memory_metric = next(
        (
            m
            for m in run["data"].get("metrics", [])
            if m.get("name") == "infer.peak_memory_usage"
        ),
        None,
    )

    if memory_metric and memory_metric.get("value"):
        from .base import create_gauge_chart

        st.markdown("#### 💾 显存使用")
        value = memory_metric["value"]
        unit = memory_metric.get("unit", "GB")

        # Try to get max memory from environment
        max_value = value * 1.5
        try:
            env = run["data"].get("environment", {})
            acc = env["cluster"][0]["machine"]["accelerators"][0]
            max_value = float(acc.get("memory_gb_per_card", 80))
        except:
            pass

        fig = create_gauge_chart(value, max_value, "峰值显存使用", "green", unit)
        st.plotly_chart(fig, use_container_width=True)
