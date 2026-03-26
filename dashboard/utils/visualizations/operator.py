#!/usr/bin/env python3
"""Operator-specific visualization functions."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from .base import create_gauge_chart


def extract_operator_metrics(data: dict) -> dict:
    """Extract operator performance metrics."""
    metrics = data.get("metrics", [])

    result = {
        "latency": None,
        "flops": None,
        "bandwidth": None,
        "accuracy": None,
    }

    for m in metrics:
        name = m.get("name", "").lower()
        value = m.get("value")

        if "latency" in name and value is not None:
            result["latency"] = float(value)
        elif "flops" in name and value is not None:
            result["flops"] = float(value)
        elif "bandwidth" in name and value is not None:
            result["bandwidth"] = float(value)
        elif "accuracy" in name or "tensor_accuracy" in name:
            result["accuracy"] = value

    return result


def render_operator_performance_charts(selected_runs, y_log, show_performance_charts):
    """Render operator performance charts (simplified like communication page)."""

    if len(selected_runs) == 1:
        run = selected_runs[0]
        metrics_data = extract_operator_metrics(run["data"])

        if metrics_data:
            st.markdown("#### 📊 核心指标")
            cols = st.columns(3)

            if metrics_data.get("latency"):
                cols[0].metric("延迟", f"{metrics_data['latency']:.2f} ms", help="算子执行延迟")
            if metrics_data.get("flops"):
                cols[1].metric(
                    "计算性能", f"{metrics_data['flops']:.2f} TFLOPS", help="每秒浮点运算次数"
                )
            if metrics_data.get("bandwidth"):
                cols[2].metric(
                    "带宽", f"{metrics_data['bandwidth']:.2f} GB/s", help="内存带宽"
                )

            # dashboard
            if show_performance_charts:
                gauge_cols = st.columns(3)

                with gauge_cols[0]:
                    if metrics_data.get("latency"):
                        max_latency = max(metrics_data["latency"] * 2, 100)
                        fig = create_gauge_chart(
                            metrics_data["latency"], max_latency, "延迟", "red", "ms"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                with gauge_cols[1]:
                    if metrics_data.get("flops"):
                        max_flops = max(metrics_data["flops"] * 1.5, 10)
                        fig = create_gauge_chart(
                            metrics_data["flops"], max_flops, "计算性能", "blue", "TFLOPS"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                with gauge_cols[2]:
                    if metrics_data.get("bandwidth"):
                        max_bandwidth = max(metrics_data["bandwidth"] * 1.5, 100)
                        fig = create_gauge_chart(
                            metrics_data["bandwidth"],
                            max_bandwidth,
                            "带宽",
                            "green",
                            "GB/s",
                        )
                        st.plotly_chart(fig, use_container_width=True)

            # Display operator configuration information
            st.markdown("#### 🔧 算子配置")
            config = run["data"].get("config", {})
            inputs = config.get("inputs", [])
            if inputs:
                st.markdown("**输入张量**")
                for inp in inputs:
                    shape = inp.get("shape", [])
                    dtype = inp.get("dtype", "unknown")
                    st.write(
                        f"- {inp.get('name', 'input')}: shape={shape}, dtype={dtype}"
                    )

            outputs = config.get("outputs", [])
            if outputs:
                st.markdown("**输出张量**")
                for out in outputs:
                    shape = out.get("shape", [])
                    dtype = out.get("dtype", "unknown")
                    st.write(
                        f"- {out.get('name', 'output')}: shape={shape}, dtype={dtype}"
                    )

    else:
        st.markdown("#### 📊 性能对比")

        all_metrics = []
        for run in selected_runs:
            metrics = extract_operator_metrics(run["data"])
            if metrics:
                config = run["data"].get("config", {})
                op_name = config.get("operator", run.get("operation", "unknown"))
                all_metrics.append(
                    {
                        "运行": f"{op_name} ({run.get('device_used', '?')}设备)",
                        "延迟 (ms)": metrics.get("latency", 0),
                        "计算性能 (TFLOPS)": metrics.get("flops", 0),
                        "带宽 (GB/s)": metrics.get("bandwidth", 0),
                    }
                )

        if all_metrics:
            df = pd.DataFrame(all_metrics)
            st.dataframe(df, use_container_width=True, hide_index=True)

            fig = go.Figure()
            for metric in ["延迟 (ms)", "计算性能 (TFLOPS)", "带宽 (GB/s)"]:
                fig.add_trace(
                    go.Bar(
                        name=metric,
                        x=[m["运行"] for m in all_metrics],
                        y=[m[metric] for m in all_metrics],
                        text=[f"{m[metric]:.2f}" for m in all_metrics],
                        textposition="auto",
                    )
                )

            fig.update_layout(
                title="算子性能对比",
                barmode="group",
                template="plotly_white",
                height=500,
                yaxis_title="性能指标",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("无法提取性能指标进行对比")
