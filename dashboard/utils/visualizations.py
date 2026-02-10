#!/usr/bin/env python3
"""Visualization functions for InfiniMetrics dashboard."""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import streamlit as st

from .data_loader import get_friendly_size


def plot_bandwidth_vs_size(df: pd.DataFrame, title: str = "带宽 vs 数据大小", 
                          y_log_scale: bool = False) -> go.Figure:
    """Plot bandwidth vs size with log scale on x-axis."""
    fig = go.Figure()
    
    # Add friendly size column for hover
    df = df.copy()
    df['size_friendly'] = df['size_bytes'].apply(get_friendly_size)
    
    # Add bandwidth line
    fig.add_trace(go.Scatter(
        x=df["size_bytes"],
        y=df["bandwidth_gbs"],
        mode="lines+markers",
        name="Bandwidth (GB/s)",
        line=dict(color="royalblue", width=3),
        marker=dict(size=8),
        hovertext=df['size_friendly'],
        hoverinfo="text+y+x"
    ))
    
    # Update layout
    layout = {
        "title": title,
        "xaxis_title": "Data Size",
        "yaxis_title": "Bandwidth (GB/s)",
        "xaxis_type": "log",
        "template": "plotly_white",
        "hovermode": "x unified",
        "height": 500
    }
    
    if y_log_scale:
        layout["yaxis_type"] = "log"
    
    fig.update_layout(**layout)
    
    # Add grid
    fig.update_xaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor="LightGray",
        tickvals=df["size_bytes"].tolist(),
        ticktext=df['size_friendly'].tolist()
    )
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
    
    return fig


def plot_latency_vs_size(df: pd.DataFrame, title: str = "延迟 vs 数据大小",
                        y_log_scale: bool = False) -> go.Figure:
    """Plot latency vs size with optional log scale."""
    fig = go.Figure()
    
    # Add friendly size column for hover
    df = df.copy()
    df['size_friendly'] = df['size_bytes'].apply(get_friendly_size)
    
    # Add latency line
    fig.add_trace(go.Scatter(
        x=df["size_bytes"],
        y=df["latency_us"],
        mode="lines+markers",
        name="Latency (µs)",
        line=dict(color="firebrick", width=3),
        marker=dict(size=8),
        hovertext=df['size_friendly'],
        hoverinfo="text+y+x"
    ))
    
    # Update layout
    layout = {
        "title": title,
        "xaxis_title": "Data Size",
        "yaxis_title": "Latency (microseconds)",
        "xaxis_type": "log",
        "template": "plotly_white",
        "hovermode": "x unified",
        "height": 500
    }
    
    if y_log_scale:
        layout["yaxis_type"] = "log"
    
    fig.update_layout(**layout)
    
    # Add grid
    fig.update_xaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor="LightGray",
        tickvals=df["size_bytes"].tolist(),
        ticktext=df['size_friendly'].tolist()
    )
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
    
    return fig


def plot_comparison_matrix(test_runs: List[Dict[str, Any]], metric: str = "bandwidth",
                          y_log_scale: bool = False) -> go.Figure:
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
            
            if metric == "bandwidth" and "bandwidth" in metric_name and metric_data.get("data") is not None:
                df = metric_data["data"].copy()
                df['size_friendly'] = df['size_bytes'].apply(get_friendly_size)
                
                # Get run info for legend
                device_used = run.get("device_used", "?")
                operation = run.get("operation", "Test")
                
                fig.add_trace(go.Scatter(
                    x=df["size_bytes"],
                    y=df["bandwidth_gbs"],
                    mode="lines+markers",
                    name=f"{operation} ({device_used} GPUs)",
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=6),
                    hovertext=df['size_friendly'],
                    hoverinfo="text+y+name"
                ))
                break  # Found bandwidth metric
            
            elif metric == "latency" and "latency" in metric_name and metric_data.get("data") is not None:
                df = metric_data["data"].copy()
                df['size_friendly'] = df['size_bytes'].apply(get_friendly_size)
                
                device_used = run.get("device_used", "?")
                operation = run.get("operation", "Test")
                
                fig.add_trace(go.Scatter(
                    x=df["size_bytes"],
                    y=df["latency_us"],
                    mode="lines+markers",
                    name=f"{operation} ({device_used} GPUs)",
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=6),
                    hovertext=df['size_friendly'],
                    hoverinfo="text+y+name"
                ))
                break  # Found latency metric
    
    metric_title = "带宽 (GB/s)" if metric == "bandwidth" else "延迟 (µs)"
    layout = {
        "title": f"多测试对比 - {metric_title}",
        "xaxis_title": "Data Size",
        "yaxis_title": metric_title,
        "xaxis_type": "log",
        "template": "plotly_white",
        "hovermode": "x unified",
        "height": 600,
        "legend": dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    }
    
    if y_log_scale:
        layout["yaxis_type"] = "log"
    
    fig.update_layout(**layout)
    
    # Set x-axis tick labels
    if test_runs and len(test_runs[0].get("data", {}).get("metrics", [])) > 0:
        first_metric = test_runs[0]["data"]["metrics"][0]
        if first_metric.get("data") is not None:
            df = first_metric["data"]
            fig.update_xaxes(
                tickvals=df["size_bytes"].tolist(),
                ticktext=df["size_bytes"].apply(get_friendly_size).tolist()
            )
    
    return fig


def create_summary_table(test_result: Dict[str, Any]) -> pd.DataFrame:
    """Create summary table from test result."""
    summary_data = []
    
    # Hardware summary
    if "environment" in test_result:
        env = test_result["environment"]
        if "cluster" in env and len(env["cluster"]) > 0:
            machine = env["cluster"][0]["machine"]
            accelerators = machine.get("accelerators", [])
            if accelerators:
                acc = accelerators[0]
                summary_data.append({"指标": "GPU型号", "数值": acc.get("model", "Unknown")})
                summary_data.append({"指标": "GPU数量", "数值": acc.get("count", "Unknown")})
                summary_data.append({"指标": "显存/卡", "数值": f"{acc.get('memory_gb_per_card', 'Unknown')} GB"})
                summary_data.append({"指标": "CUDA版本", "数值": acc.get("cuda", "Unknown")})
    
    # Test config summary
    config = test_result.get("config", {})
    resolved = test_result.get("resolved", {})
    
    # Device info
    device_used = resolved.get("device_used") or config.get("device_used") or config.get("device_involved", "Unknown")
    nodes = resolved.get("nodes") or config.get("nodes", 1)
    
    summary_data.append({"指标": "算子", "数值": config.get("operator", "Unknown")})
    summary_data.append({"指标": "设备数", "数值": device_used})
    summary_data.append({"指标": "节点数", "数值": nodes})
    summary_data.append({"指标": "预热迭代", "数值": config.get("warmup_iterations", "Unknown")})
    summary_data.append({"指标": "测量迭代", "数值": config.get("measured_iterations", "Unknown")})
    
    # Performance summary (extract from metrics if available)
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
    
    # Duration
    duration = next((m["value"] for m in test_result.get("metrics", []) 
                    if m.get("name") == "comm.duration"), None)
    if duration:
        summary_data.append({"指标": "测试耗时", "数值": f"{duration:.2f} ms"})
    
    return pd.DataFrame(summary_data)


def create_gauge_chart(value: float, max_value: float, title: str, 
                      color: str = "blue", unit: str = "") -> go.Figure:
    """Create a gauge chart for single metric visualization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{title}<br>{unit}"},
        number={'suffix': f" {unit}"},
        gauge={
            'axis': {'range': [0, max_value]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, max_value * 0.6], 'color': "lightgray"},
                {'range': [max_value * 0.6, max_value * 0.8], 'color': "gray"},
                {'range': [max_value * 0.8, max_value], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.9
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(t=50, b=10, l=10, r=10))
    return fig

def plot_timeseries_auto(df: pd.DataFrame, title: str = "Timeseries", y_log_scale: bool = False) -> go.Figure:
    """
    Generic plot for 2-column timeseries CSV:
    - infer: (timestamp, latency_ms/ttft_ms/throughput)
    - future ops: (step, value) etc.
    """
    fig = go.Figure()
    if df is None or df.empty or len(df.columns) < 2:
        fig.update_layout(title=f"{title} (no data)")
        return fig

    xcol = df.columns[0]
    ycol = df.columns[1]

    fig.add_trace(go.Scatter(
        x=df[xcol],
        y=df[ycol],
        mode="lines+markers",
        name=ycol
    ))

    fig.update_layout(
        title=title,
        xaxis_title=str(xcol),
        yaxis_title=str(ycol),
        template="plotly_white",
        height=420,
        hovermode="x unified",
    )
    if y_log_scale:
        fig.update_yaxes(type="log")
    return fig


def create_summary_table_infer(test_result: dict) -> pd.DataFrame:
    rows = []

    # env brief
    env = test_result.get("environment", {})
    try:
        acc = env["cluster"][0]["machine"]["accelerators"][0]
        rows += [
            {"指标": "加速卡", "数值": acc.get("model", "Unknown")},
            {"指标": "卡数", "数值": acc.get("count", "Unknown")},
            {"指标": "显存/卡", "数值": f"{acc.get('memory_gb_per_card','?')} GB"},
            {"指标": "CUDA", "数值": acc.get("cuda", "Unknown")},
            {"指标": "平台", "数值": acc.get("type", "nvidia")},
        ]
    except Exception:
        pass

    cfg = test_result.get("config", {})
    rows += [
        {"指标": "框架", "数值": cfg.get("framework", "unknown")},
        {"指标": "模型", "数值": cfg.get("model", "")},
        {"指标": "batch", "数值": (cfg.get("infer_args", {}) or {}).get("static_batch_size", "unknown")},
        {"指标": "prompt_tok", "数值": (cfg.get("infer_args", {}) or {}).get("prompt_token_num", "unknown")},
        {"指标": "output_tok", "数值": (cfg.get("infer_args", {}) or {}).get("output_token_num", "unknown")},
        {"指标": "warmup", "数值": cfg.get("warmup_iterations", "unknown")},
        {"指标": "measured", "数值": cfg.get("measured_iterations", "unknown")},
    ]

    # scalar metrics quick view
    for m in test_result.get("metrics", []):
        if m.get("type") == "scalar":
            rows.append({"指标": m.get("name"), "数值": f"{m.get('value')} {m.get('unit','')}"})

    return pd.DataFrame(rows)


def create_summary_table_ops(test_result: dict) -> pd.DataFrame:
    rows = []
    cfg = test_result.get("config", {})

    rows.append({"指标": "testcase", "数值": test_result.get("testcase", "")})
    # Try to get operator name from config
    rows.append({"指标": "算子", "数值": cfg.get("operator", cfg.get("op_name", "Unknown"))})

    # Environment info
    env = test_result.get("environment", {})
    try:
        acc = env["cluster"][0]["machine"]["accelerators"][0]
        rows += [
            {"指标": "加速卡", "数值": acc.get("model", "Unknown")},
            {"指标": "卡数", "数值": acc.get("count", "Unknown")},
        ]
    except Exception:
        pass

    # Scalar metrics summary
    scalars = [m for m in test_result.get("metrics", []) if m.get("type") == "scalar"]
    for m in scalars:
        rows.append({"指标": m.get("name"), "数值": f"{m.get('value')} {m.get('unit','')}"})

    # Common config fields fallback
    for k in ["dtype", "shape", "batch_size", "warmup_iterations", "measured_iterations"]:
        if k in cfg:
            rows.append({"指标": k, "数值": cfg.get(k)})

    return pd.DataFrame(rows)
