#!/usr/bin/env python3
"""CI/CD Test history page"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import logging

from common import init_page, show_data_source_info
from components.header import render_header

# Configure logger
logger = logging.getLogger(__name__)

init_page("CI历史记录 | InfiniMetrics", "📋")


def main():
    render_header()

    st.markdown(
        """
    <div style="text-align: center; margin-bottom: 1rem;">
        <h1>📋 CI/CD 测试历史记录</h1>
        <p style="color: #666; font-size: 1rem;">
            展示每次CI运行的测试汇总结果，包括测试套件执行情况和失败详情
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Get data loader
    dl = st.session_state.data_loader

    # Load CI history data
    with st.spinner("加载CI历史数据..."):
        ci_history = dl.load_ci_history(limit=200)

    if not ci_history:
        st.info(
            """
        ### 📭 暂无CI历史记录
        
        可能的原因：
        - 还没有CI任务运行完成
        - summary_output目录下没有摘要文件
        
        **当前数据源:** {}
        """.format(
                "MongoDB"
                if dl.is_using_mongodb
                else f"文件系统 ({dl.results_dir.parent}/summary_output)"
            )
        )
        return

    # Convert to DataFrame
    df = create_history_dataframe(ci_history)

    if df.empty:
        st.warning("没有有效的CI记录数据")
        return

    # Sidebar filter conditions
    with st.sidebar:
        st.markdown("## 🔍 筛选条件")
        show_data_source_info(style="sidebar")
        st.markdown("---")

        # Initialize date range variable
        date_range = None

        # Date range filter
        if not df.empty and "日期" in df.columns:
            min_date = df["日期"].min().date()
            max_date = df["日期"].max().date()
            date_range = st.date_input(
                "日期范围",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )

        # Status filter
        if "状态" in df.columns:
            status_options = sorted(df["状态"].unique())
            status_filter = st.multiselect(
                "运行状态", options=status_options, default=status_options
            )
        else:
            status_filter = None

        # Branch filter
        if "分支" in df.columns:
            branches = sorted(df["分支"].unique())
            branch_filter = st.multiselect("分支", options=branches, default=branches)
        else:
            branch_filter = None

        # Commit message search
        commit_search = st.text_input("🔎 提交信息搜索", placeholder="输入commit ID或提交信息")

        # Show only failed runs
        show_only_failed = st.checkbox("仅显示失败的运行", value=False)

    # Apply filters
    filtered_df = apply_filters(
        df, date_range, status_filter, branch_filter, commit_search, show_only_failed
    )

    if filtered_df.empty:
        st.warning("没有符合条件的CI记录")
        return

    # Render overview metrics
    render_overview_metrics(filtered_df)

    # Render trend chart
    render_trend_chart(filtered_df)

    # Render detailed table
    render_detailed_table(filtered_df)

    # If there are failed runs, show failure details
    render_failure_details(filtered_df)


def create_history_dataframe(ci_history: list) -> pd.DataFrame:
    """Convert CI history to DataFrame"""
    rows = []
    for item in ci_history:
        try:
            # Parse timestamp
            timestamp = item.get("timestamp", "")
            try:
                if isinstance(timestamp, str):
                    if "_" in timestamp and len(timestamp) == 15:
                        dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                    else:
                        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                else:
                    dt = datetime.now()
            except Exception as e:
                logger.warning(f"解析时间戳失败: {timestamp}, 错误: {e}")
                dt = datetime.now()

            # Get Git information
            short_commit = item.get("short_commit", "")
            commit = item.get("commit", "")
            if short_commit and short_commit != "unknown":
                commit_display = short_commit[:8]
            elif commit and commit != "unknown":
                commit_display = commit[:8]
            else:
                commit_display = "📁 本地运行"

            branch = item.get("branch", "")
            if not branch or branch == "unknown" or branch.startswith("unknown"):
                branch_display = "local"
            else:
                branch_display = branch

            commit_msg = item.get("commit_message", "")
            if not commit_msg or commit_msg.startswith("unknown"):
                msg_display = "未在Git仓库中运行"
            else:
                msg_display = (
                    commit_msg[:60] + "..." if len(commit_msg) > 60 else commit_msg
                )

            commit_author = item.get("commit_author", "unknown")
            if commit_author == "unknown" or commit_author.startswith("unknown"):
                commit_author = "-"

            # Get test statistics
            total = item.get("total_tests", 0)
            success = item.get("successful_tests", 0)
            failed = item.get("failed_tests", 0)
            success_rate = (success / total * 100) if total > 0 else 0
            duration = item.get("duration", item.get("total_duration_seconds", 0))

            # Get status
            status = item.get("status", "未知")

            # Extract all run_ids (from results)
            results = item.get("results", [])
            run_ids = [r.get("run_id", "unknown") for r in results if r.get("run_id")]
            run_id_display = ", ".join(run_ids) if run_ids else "unknown"

            # Extract failed test details
            failed_details = []
            for result in results:
                if result.get("result_code", 0) != 0:
                    failed_details.append(
                        {
                            "testcase": result.get("testcase", "unknown"),
                            "run_id": result.get("run_id", "unknown"),
                            "result_code": result.get("result_code", -1),
                            "result_file": result.get("result_file", ""),
                        }
                    )

            rows.append(
                {
                    "时间": dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "日期": dt,
                    "Run ID": run_id_display,
                    "Commit": commit_display,
                    "分支": branch_display,
                    "提交信息": msg_display,
                    "作者": commit_author,
                    "总测试数": total,
                    "成功": success,
                    "失败": failed,
                    "成功率": f"{success_rate:.1f}%",
                    "耗时": f"{duration:.1f}s" if duration > 0 else "-",
                    "状态": status,
                    "失败详情": failed_details,
                    "_raw": item,
                }
            )
        except Exception as e:
            logger.warning(f"解析记录失败: {e}")
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "日期" in df.columns:
        df["日期"] = pd.to_datetime(df["日期"])

    return df


def apply_filters(
    df, date_range, status_filter, branch_filter, commit_search, show_only_failed
):
    """Apply a filter condition"""
    if df.empty:
        return df

    filtered = df.copy()

    # Date filter
    if date_range and len(date_range) == 2 and "日期" in filtered.columns:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered["日期"].dt.date >= start_date)
            & (filtered["日期"].dt.date <= end_date)
        ]

    # Status filter
    if status_filter and "状态" in filtered.columns:
        filtered = filtered[filtered["状态"].isin(status_filter)]

    # Branch filter
    if branch_filter and "分支" in filtered.columns:
        filtered = filtered[filtered["分支"].isin(branch_filter)]

    # Commit message search (also supports Run ID search)
    if commit_search and "提交信息" in filtered.columns:
        filtered = filtered[
            filtered["提交信息"].str.contains(commit_search, case=False, na=False)
            | filtered["Commit"].str.contains(commit_search, case=False, na=False)
            | filtered["Run ID"].str.contains(commit_search, case=False, na=False)
        ]

    # Show only failed
    if show_only_failed and "失败" in filtered.columns:
        filtered = filtered[filtered["失败"] > 0]

    # Sort by date
    if "日期" in filtered.columns and not filtered.empty:
        return filtered.sort_values("日期", ascending=False)
    return filtered


def render_overview_metrics(df, ci_history=None):
    """Overview indicators for rendering"""
    st.markdown("### 📊 概览统计")

    total_runs = len(df)
    total_tests = df["总测试数"].sum() if "总测试数" in df.columns else 0
    total_success = df["成功"].sum() if "成功" in df.columns else 0
    total_failed = df["失败"].sum() if "失败" in df.columns else 0
    avg_success_rate = (total_success / total_tests * 100) if total_tests > 0 else 0

    # Calculate trend
    recent_runs = df.head(min(10, len(df)))
    recent_success_rate = (
        (recent_runs["成功"].sum() / recent_runs["总测试数"].sum() * 100)
        if "总测试数" in recent_runs.columns and recent_runs["总测试数"].sum() > 0
        else 0
    )

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("总运行次数", total_runs)
    col2.metric("总测试数", f"{total_tests:,}")
    col3.metric("总成功", f"{total_success:,}")
    col4.metric("总失败", f"{total_failed:,}")
    col5.metric("平均成功率", f"{avg_success_rate:.1f}%")
    col6.metric(
        "最近10次成功率",
        f"{recent_success_rate:.1f}%",
        delta=f"{recent_success_rate - avg_success_rate:.1f}%",
    )

    # Data source information
    dl = st.session_state.data_loader
    if dl.is_using_mongodb:
        st.caption("🟢 数据源: MongoDB (实时CI数据)")
    else:
        st.caption(f"📁 数据源: 文件系统 - 摘要目录: {dl.results_dir.parent}/summary_output")


def render_trend_chart(df):
    """Rendering trend chart"""
    st.markdown("### 📈 历史趋势")

    # Group statistics by date
    if "日期" not in df.columns:
        st.warning("没有日期数据，无法显示趋势图表")
        return

    # Extract date
    df["日期_显示"] = df["日期"].dt.strftime("%Y-%m-%d")

    daily_stats = (
        df.groupby("日期_显示").agg({"总测试数": "sum", "成功": "sum", "失败": "sum"}).reset_index()
    )
    daily_stats.columns = ["日期", "总测试数", "成功", "失败"]

    # Test count trend
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=daily_stats["日期"],
            y=daily_stats["总测试数"],
            name="总测试数",
            marker_color="lightblue",
            text=daily_stats["总测试数"],
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            x=daily_stats["日期"],
            y=daily_stats["成功"],
            name="成功",
            marker_color="lightgreen",
            text=daily_stats["成功"],
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            x=daily_stats["日期"],
            y=daily_stats["失败"],
            name="失败",
            marker_color="lightcoral",
            text=daily_stats["失败"],
            textposition="auto",
        )
    )

    fig.update_layout(
        title="每日测试次数分布",
        barmode="group",
        xaxis_title="日期",
        yaxis_title="测试次数",
        template="plotly_white",
        height=450,
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_detailed_table(df):
    """Render the detailed record table"""
    st.markdown("### 📋 详细记录")

    # Check which columns actually exist in DataFrame
    available_cols = df.columns.tolist()

    # Define columns to display
    display_cols = []
    for col in [
        "时间",
        "Run ID",
        "Commit",
        "提交信息",
        "分支",
        "作者",
        "总测试数",
        "成功",
        "失败",
        "成功率",
        "状态",
        "耗时",
    ]:
        if col in available_cols:
            display_cols.append(col)

    if not display_cols:
        st.warning("没有可显示的列")
        return

    display_df = df[display_cols].copy()

    # Add color styling
    def color_status(val):
        if "成功" in str(val):
            return "background-color: #d4edda; color: #155724"
        elif "失败" in str(val):
            return "background-color: #f8d7da; color: #721c24"
        elif "部分成功" in str(val):
            return "background-color: #fff3cd; color: #856404"
        return ""

    if "状态" in display_df.columns:
        styled_df = display_df.style.applymap(color_status, subset=["状态"])
    else:
        styled_df = display_df

    # Add color styling
    column_config = {
        "时间": st.column_config.TextColumn("时间", width="small"),
        "Run ID": st.column_config.TextColumn("Run ID", width="medium"),
        "Commit": st.column_config.TextColumn("Commit", width="small"),
        "提交信息": st.column_config.TextColumn("提交信息", width="large"),
        "分支": st.column_config.TextColumn("分支", width="small"),
        "作者": st.column_config.TextColumn("作者", width="small"),
        "总测试数": st.column_config.NumberColumn("总数", width="small"),
        "成功": st.column_config.NumberColumn("✅", width="small"),
        "失败": st.column_config.NumberColumn("❌", width="small"),
        "成功率": st.column_config.TextColumn("成功率", width="small"),
        "状态": st.column_config.TextColumn("状态", width="small"),
        "耗时": st.column_config.TextColumn("时长", width="small"),
    }

    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        column_config=column_config,
    )


def render_failure_details(df):
    """Details of rendering failure"""
    if "失败" not in df.columns:
        return

    failed_runs = df[df["失败"] > 0]
    if failed_runs.empty:
        st.success("🎉 所有运行均无失败测试！")
        return

    st.markdown("### 🔍 失败详情")
    st.caption("点击展开查看失败测试的详细信息")

    for _, row in failed_runs.iterrows():
        failed_details = row.get("失败详情", [])

        if not failed_details:
            with st.expander(f"📅 {row['时间']} - 失败: {row['失败']}个测试"):
                st.warning(f"该次运行有 {row['失败']} 个测试失败，但无详细错误信息")
                if "Run ID" in row:
                    st.caption(f"Run ID: {row['Run ID']}")
            continue

        with st.expander(f"📅 {row['时间']} - 失败: {len(failed_details)}个测试"):
            for i, fail in enumerate(failed_details[:20]):
                st.markdown(f"**{i+1}. {fail.get('testcase', 'unknown')}**")
                st.markdown(f"- Run ID: `{fail.get('run_id', 'unknown')}`")
                st.markdown(f"- Result Code: {fail.get('result_code', -1)}")
                if fail.get("result_file"):
                    st.markdown(f"- Result File: `{fail.get('result_file')}`")
                st.divider()

            if len(failed_details) > 20:
                st.info(f"还有 {len(failed_details) - 20} 个失败测试未显示")


if __name__ == "__main__":
    main()
