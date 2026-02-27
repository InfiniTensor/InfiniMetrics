#!/usr/bin/env python3
"""Main Streamlit application for InfiniMetrics dashboard."""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime
from infinimetrics.common.constants import AcceleratorType

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from components.header import render_header
from utils.data_loader import InfiniMetricsDataLoader, load_summary_file

# Page configuration
st.set_page_config(
    page_title="InfiniMetrics Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "data_loader" not in st.session_state:
    st.session_state.data_loader = InfiniMetricsDataLoader()
if "selected_accelerators" not in st.session_state:
    st.session_state.selected_accelerators = []


def main():
    render_header()

    # =========================
    # Sidebar
    # =========================

    with st.sidebar:
        st.markdown("## ⚙️ 设置")

        results_dir = st.text_input(
            "测试结果目录", value="./test_output", help="包含 JSON/CSV 测试结果的目录"
        )

        if results_dir != str(st.session_state.data_loader.results_dir):
            st.session_state.data_loader = InfiniMetricsDataLoader(results_dir)

        auto_refresh = st.toggle("自动刷新", value=False)
        if auto_refresh:
            st.rerun()

        st.markdown("---")
        st.markdown("## 🧠 筛选条件")

        # Base accelerator types from constants.py
        ACCELERATOR_OPTIONS = ["cpu"] + [a.value for a in AcceleratorType]

        # UI display names (only labels live here)
        ACCELERATOR_LABELS = {
            "cpu": "CPU",
            AcceleratorType.NVIDIA.value: "NVIDIA",
            AcceleratorType.AMD.value: "AMD",
            AcceleratorType.ASCEND.value: "昇腾 NPU",
            AcceleratorType.CAMBRICON.value: "寒武纪 MLU",
            AcceleratorType.GENERIC.value: "Generic",
        }

        selected_accs = st.multiselect(
            "加速卡类型",
            options=ACCELERATOR_OPTIONS,
            default=ACCELERATOR_OPTIONS,
            format_func=lambda x: ACCELERATOR_LABELS.get(x, x),
        )
        st.session_state.selected_accelerators = selected_accs

        run_id_filter = st.text_input("Run ID 模糊搜索")
        # test_type / testcase filtering will be applied dynamically after runs are loaded

    render_dashboard(run_id_filter)


def render_dashboard(run_id_filter: str):
    st.markdown(
        """
        <h1 style="margin-bottom: 0.2em;">
            📊 综合仪表板
        </h1>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="
            margin-top: 0.5em;
            margin-bottom: 1.5em;
            max-width: 1100px;
            font-size: 1.05em;
            line-height: 1.6;
        ">
            <strong>InfiniMetrics Dashboard</strong> 用于统一展示
            <strong>通信（NCCL / 集合通信）</strong>、
            <strong>推理（Direct / Service）</strong>、
            <strong>算子（核心算子性能）</strong>
            等 AI 加速卡性能测试结果。
            <br/>
            测试框架输出 <code>JSON</code>（环境 / 配置 / 标量指标） +
            <code>CSV</code>（曲线 / 时序数据），
            Dashboard 自动加载并支持多次运行的对比分析与可视化。
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        runs = st.session_state.data_loader.list_test_runs()

        # ========== Accelerator filtering  ==========
        selected_accs = st.session_state.get("selected_accelerators", [])
        if selected_accs:
            runs = [
                r
                for r in runs
                if set(r.get("accelerator_types", [])) & set(selected_accs)
            ]

        # ========== run_id filtering  ==========
        if run_id_filter:
            runs = [r for r in runs if run_id_filter in r.get("run_id", "")]

        if not runs:
            st.warning("No test results match the current filters.")
            return

        # ========== Sort by time (latest first) ==========
        def _parse_time(t):
            try:
                return datetime.fromisoformat(t)
            except Exception:
                return datetime.min

        runs = sorted(runs, key=lambda r: _parse_time(r.get("time", "")), reverse=True)

        total = len(runs)
        success = sum(1 for r in runs if r.get("success"))
        fail = total - success

        # ========== Categorize runs ==========
        comm_runs = [r for r in runs if r.get("testcase", "").startswith("comm")]
        infer_runs = [r for r in runs if r.get("testcase", "").startswith("infer")]

        ops_runs, hw_runs = [], []
        for r in runs:
            p = str(r.get("path", "")).replace("\\", "/").lower()
            tc = (r.get("testcase", "") or "").lower()
            if "/operators/" in p or tc.startswith(("operator", "operators", "ops")):
                ops_runs.append(r)
            if "/hardware/" in p or tc.startswith("hardware"):
                hw_runs.append(r)

        # ========== KPI ==========
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("总测试数", total)
        c2.metric("成功率", f"{(success/total*100):.1f}%")
        c3.metric("通信测试", len(comm_runs))
        c4.metric("推理测试", len(infer_runs))
        c5.metric("算子测试", len(ops_runs))
        c6.metric("硬件检测", len(hw_runs))

        st.caption(f"失败测试数：{fail}")
        st.caption(f"当前筛选：加速卡={','.join(selected_accs) or '全部'}")

        st.divider()

        # ========== Latest results ==========
        def _latest(lst):
            return lst[0] if lst else None

        latest_comm = _latest(comm_runs)
        latest_infer = _latest(infer_runs)
        latest_ops = _latest(ops_runs)

        colA, colB, colC = st.columns(3)

        with colA:
            st.markdown("#### 🔗 通信（最新）")
            if not latest_comm:
                st.info("暂无通信结果")
            else:
                st.write(f"- testcase: `{latest_comm.get('testcase','')}`")
                st.write(f"- time: {latest_comm.get('time','')}")
                st.write(f"- status: {'✅' if latest_comm.get('success') else '❌'}")

        with colB:
            st.markdown("#### 🚀 推理（最新）")
            if not latest_infer:
                st.info("暂无推理结果")
            else:
                st.write(f"- testcase: `{latest_infer.get('testcase','')}`")
                st.write(f"- time: {latest_infer.get('time','')}")
                st.write(f"- status: {'✅' if latest_infer.get('success') else '❌'}")

        with colC:
            st.markdown("#### ⚡ 算子（最新）")
            if not latest_ops:
                st.info("暂无算子结果")
            else:
                st.write(f"- testcase: `{latest_ops.get('testcase','')}`")
                st.write(f"- time: {latest_ops.get('time','')}")
                st.write(f"- status: {'✅' if latest_ops.get('success') else '❌'}")

        st.divider()

        # ========== Recent runs table ==========
        st.markdown("### 🕒 最近测试运行")
        df = pd.DataFrame(
            [
                {
                    "类型": (r.get("testcase", "").split(".")[0] or "UNKNOWN").upper(),
                    "加速卡": ", ".join(r.get("accelerator_types", [])),
                    "时间": r.get("time", ""),
                    "状态": "✅" if r.get("success") else "❌",
                    "run_id": r.get("run_id", "")[:32],
                }
                for r in runs[:15]
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

        # ========== Dispatcher summary ==========
        summaries = load_summary_file()

        if not summaries:
            st.info("No dispatcher_summary file found")
            return

        st.markdown("### 🧾 Dispatcher 汇总记录")

        rows = []
        for s in summaries:
            rows.append(
                {
                    "时间": s.get("timestamp"),
                    "总测试数": s.get("total_tests"),
                    "成功": s.get("successful_tests"),
                    "失败": s.get("failed_tests"),
                    "成功率": (
                        f"{s['successful_tests'] / s['total_tests'] * 100:.1f}%"
                        if s.get("total_tests")
                        else "-"
                    ),
                    "文件": s.get("file"),
                }
            )

        df = pd.DataFrame(rows).sort_values("时间", ascending=False)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

        # ========== Quick navigation ==========
        st.markdown("---")
        st.markdown("### 🚀 快速导航")

        col1, col2, col3 = st.columns(3)
        if col1.button("🔗 通信测试分析", use_container_width=True):
            st.switch_page("pages/communication.py")
        if col2.button("⚡ 算子测试分析", use_container_width=True):
            st.switch_page("pages/operator.py")
        if col3.button("🤖 推理测试分析", use_container_width=True):
            st.switch_page("pages/inference.py")

    except Exception as e:
        st.error(f"Dashboard 加载失败: {e}")


if __name__ == "__main__":
    main()
