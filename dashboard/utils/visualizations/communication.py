#!/usr/bin/env python3
"""Communication-specific visualization functions."""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, List, Any, Optional, Literal
import streamlit as st

from utils.data_loader import get_friendly_size


def plot_metric_vs_size(
    df: pd.DataFrame,
    metric_type: Literal["bandwidth", "latency"],
    title: Optional[str] = None,
    y_log_scale: bool = False,
) -> go.Figure:
    """Generic plot for metric vs message size."""
    fig = go.Figure()

    metric_configs = {
        "bandwidth": {
            "y_column": "bandwidth_gbs",
            "y_title": "Bandwidth (GB/s)",
            "line_color": "royalblue",
            "name": "Bandwidth",
            "default_title": "带宽 vs 数据大小",
        },
        "latency": {
            "y_column": "latency_us",
            "y_title": "Latency (microseconds)",
            "line_color": "firebrick",
            "name": "Latency",
            "default_title": "延迟 vs 数据大小",
        },
    }

    config = metric_configs.get(metric_type)
    if not config:
        raise ValueError(f"Unsupported metric_type: {metric_type}")

    if config["y_column"] not in df.columns:
        st.warning(f"DataFrame missing required column: {config['y_column']}")
        fig.update_layout(title=f"{title or config['default_title']} (no data)")
        return fig

    df = df.copy()
    df["size_friendly"] = df["size_bytes"].apply(get_friendly_size)

    fig.add_trace(
        go.Scatter(
            x=df["size_bytes"],
            y=df[config["y_column"]],
            mode="lines+markers",
            name=config["name"],
            line=dict(color=config["line_color"], width=3),
            marker=dict(size=8),
            hovertext=df["size_friendly"],
            hoverinfo="text+y+x",
        )
    )

    layout = {
        "title": title or config["default_title"],
        "xaxis_title": "Data Size",
        "yaxis_title": config["y_title"],
        "xaxis_type": "log",
        "template": "plotly_white",
        "hovermode": "x unified",
        "height": 500,
    }

    if y_log_scale:
        layout["yaxis_type"] = "log"

    fig.update_layout(**layout)

    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="LightGray",
        tickvals=df["size_bytes"].tolist(),
        ticktext=df["size_friendly"].tolist(),
    )
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

    return fig


def plot_comparison_matrix(
    test_runs: List[Dict[str, Any]],
    metric: str = "bandwidth",
    y_log_scale: bool = False,
) -> go.Figure:
    """Create comparison matrix for multiple test runs."""
    fig = go.Figure()
    colors = px.colors.qualitative.Set2

    for i, run in enumerate(test_runs):
        if i >= len(colors):
            break

        data = run.get("data", {})
        metrics = data.get("metrics", [])

        for metric_data in metrics:
            metric_name = metric_data.get("name", "")

            if (
                metric == "bandwidth"
                and "bandwidth" in metric_name
                and metric_data.get("data") is not None
            ):
                df = metric_data["data"].copy()
                df["size_friendly"] = df["size_bytes"].apply(get_friendly_size)

                device_used = run.get("device_used", "?")
                operation = run.get("operation", "Test")

                fig.add_trace(
                    go.Scatter(
                        x=df["size_bytes"],
                        y=df["bandwidth_gbs"],
                        mode="lines+markers",
                        name=f"{operation} ({device_used} GPUs)",
                        line=dict(color=colors[i % len(colors)], width=2),
                        marker=dict(size=6),
                        hovertext=df["size_friendly"],
                        hoverinfo="text+y+name",
                    )
                )
                break

            elif (
                metric == "latency"
                and "latency" in metric_name
                and metric_data.get("data") is not None
            ):
                df = metric_data["data"].copy()
                df["size_friendly"] = df["size_bytes"].apply(get_friendly_size)

                device_used = run.get("device_used", "?")
                operation = run.get("operation", "Test")

                fig.add_trace(
                    go.Scatter(
                        x=df["size_bytes"],
                        y=df["latency_us"],
                        mode="lines+markers",
                        name=f"{operation} ({device_used} GPUs)",
                        line=dict(color=colors[i % len(colors)], width=2),
                        marker=dict(size=6),
                        hovertext=df["size_friendly"],
                        hoverinfo="text+y+name",
                    )
                )
                break

    metric_title = "带宽 (GB/s)" if metric == "bandwidth" else "延迟 (µs)"
    layout = {
        "title": f"多测试对比 - {metric_title}",
        "xaxis_title": "Data Size",
        "yaxis_title": metric_title,
        "xaxis_type": "log",
        "template": "plotly_white",
        "hovermode": "x unified",
        "height": 600,
        "legend": dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    }

    if y_log_scale:
        layout["yaxis_type"] = "log"

    fig.update_layout(**layout)

    if test_runs and len(test_runs[0].get("data", {}).get("metrics", [])) > 0:
        first_metric = test_runs[0]["data"]["metrics"][0]
        if first_metric.get("data") is not None:
            df = first_metric["data"]
            fig.update_xaxes(
                tickvals=df["size_bytes"].tolist(),
                ticktext=df["size_bytes"].apply(get_friendly_size).tolist(),
            )

    return fig
