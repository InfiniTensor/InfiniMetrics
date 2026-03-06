#!/usr/bin/env python3
"""Hardware tests analysis page."""

import streamlit as st
import pandas as pd

from common import init_page
from components.header import render_header
from utils.visualizations import (
    create_summary_table_hw,
    plot_timeseries_auto,
)

init_page("硬件测试分析 | InfiniMetrics", "🔧")


def main():
    render_header()
    st.markdown("## 🔧 硬件性能测试分析")

    dl = st.session_state.data_loader

    # Debug info - show based on source type
    if dl.source_type == "mongodb":
        st.caption("数据源: MongoDB")
    else:
        st.caption(f"数据源: 文件系统 ({dl.results_dir})")

    runs = dl.list_test_runs()
    # Identify hardware runs by testcase starting with hardware
    hw_runs = [r for r in runs if (r.get("testcase") or "").startswith("hardware")]

    if not hw_runs:
        st.info("未找到硬件测试结果（testcase 需以 hardware.* 开头）。")
        return

    with st.sidebar:
        st.markdown("### 🔍 筛选条件")
        only_success = st.checkbox("仅显示成功测试", value=True)
        y_log = st.checkbox("Y轴对数刻度（可选）", value=False)

    filtered = [r for r in hw_runs if (not only_success or r.get("success"))]

    st.caption(f"找到 {len(filtered)} 个硬件测试")

    options = {
        f"{r.get('testcase','unknown')} | {r.get('time','')} | {r.get('run_id','')[:12]}": i
        for i, r in enumerate(filtered)
    }
    selected = st.multiselect(
        "选择要分析的测试运行（可多选）",
        list(options.keys()),
        default=list(options.keys())[:1],
    )
    if not selected:
        return

    selected_runs = []
    for k in selected:
        ri = filtered[options[k]]
        # Use path for file source, run_id for MongoDB
        identifier = ri.get("path") or ri.get("run_id")
        data = dl.load_test_result(identifier)
        ri = dict(ri)
        ri["data"] = data
        selected_runs.append(ri)

    tab1, tab2 = st.tabs(["📌 概览", "📈 曲线/原始数据"])

    with tab1:
        for run in selected_runs:
            with st.expander(f"{run.get('run_id')} - 概览"):
                st.dataframe(
                    create_summary_table_hw(run["data"]),
                    use_container_width=True,
                    hide_index=True,
                )
                st.markdown("**config**")
                st.json(run["data"].get("config", {}))
                st.markdown("**environment**")
                st.json(run["data"].get("environment", {}))

    with tab2:
        for run in selected_runs:
            with st.expander(f"{run.get('run_id')} - metrics"):
                metrics = run["data"].get("metrics", [])

                # Group metrics by type
                mem_metrics = [
                    m for m in metrics if "mem_sweep" in m.get("name", "")
                ]
                cache_metrics = [
                    m for m in metrics if "cache" in m.get("name", "")
                ]
                stream_metrics = [
                    m for m in metrics if "stream" in m.get("name", "")
                ]

                # Memory bandwidth plots
                if mem_metrics:
                    st.markdown("#### 内存带宽 (Memory Sweep)")
                    cols = st.columns(min(3, len(mem_metrics)))
                    for i, m in enumerate(mem_metrics):
                        with cols[i % len(cols)]:
                            df = m.get("data")
                            if df is not None and len(df.columns) >= 2:
                                fig = plot_timeseries_auto(
                                    df,
                                    title=m.get("name", "memory"),
                                    y_log_scale=y_log,
                                )
                                st.plotly_chart(fig, use_container_width=True)

                # Cache bandwidth plots
                if cache_metrics:
                    st.markdown("#### 缓存带宽 (Cache)")
                    for m in cache_metrics:
                        df = m.get("data")
                        if df is not None and len(df.columns) >= 2:
                            fig = plot_timeseries_auto(
                                df,
                                title=m.get("name", "cache"),
                                y_log_scale=y_log,
                            )
                            st.plotly_chart(fig, use_container_width=True)

                # STREAM benchmark scalars
                if stream_metrics:
                    st.markdown("#### STREAM 基准测试")
                    stream_data = []
                    for m in stream_metrics:
                        stream_data.append({
                            "指标": m.get("name", ""),
                            "数值": f"{m.get('value', 0):.2f} {m.get('unit', '')}"
                        })
                    if stream_data:
                        st.dataframe(
                            pd.DataFrame(stream_data),
                            use_container_width=True,
                            hide_index=True,
                        )


if __name__ == "__main__":
    main()
