#!/usr/bin/env python3
"""Operator tests analysis page."""

import streamlit as st
import pandas as pd

from common import init_page, show_data_source_info
from components.header import render_header
from utils.visualizations import (
    create_summary_table_ops,
    plot_timeseries_auto,
)

init_page("算子测试分析 | InfiniMetrics", "⚡")


def main():
    render_header()
    st.markdown("## ⚡ 算子测试分析")

    show_data_source_info()

    runs = st.session_state.data_loader.list_test_runs()
    # Identify operator runs by testcase starting with operator/ops
    ops_runs = []
    for r in runs:
        tc = (r.get("testcase") or "").lower()
        if tc.startswith(("operator", "operators", "ops")):
            ops_runs.append(r)

    if not ops_runs:
        st.info("未找到算子测试结果（请确认 JSON 在 output/operators 目录下）。")
        return

    with st.sidebar:
        st.markdown("### 🔍 筛选条件")
        only_success = st.checkbox("仅显示成功测试", value=True)
        y_log = st.checkbox("Y轴对数刻度（可选）", value=False)

    filtered = [r for r in ops_runs if (not only_success or r.get("success"))]

    st.caption(f"找到 {len(filtered)} 个算子测试")

    options = {
        f"{r.get('operation','unknown')} | {r.get('time','')} | {r.get('run_id','')[:12]}": i
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
                    create_summary_table_ops(run["data"]),
                    use_container_width=True,
                    hide_index=True,
                )
                st.markdown("**config**")
                st.json(run["data"].get("config", {}))

    with tab2:
        # If operators have timeseries CSVs, automatically plot them
        for run in selected_runs:
            with st.expander(f"{run.get('run_id')} - metrics"):
                for m in run["data"].get("metrics", []):
                    df = m.get("data")
                    if df is not None and len(df.columns) >= 2:
                        fig = plot_timeseries_auto(
                            df, title=m.get("name", "metric"), y_log_scale=y_log
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # scalar
                        if m.get("type") == "scalar":
                            st.write(
                                f"- {m.get('name')}: {m.get('value')} {m.get('unit','')}"
                            )


if __name__ == "__main__":
    main()
