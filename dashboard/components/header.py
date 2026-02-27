#!/usr/bin/env python3
"""Header component for InfiniMetrics dashboard."""

import streamlit as st


def render_header():
    """Render the dashboard header."""
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1E3A8A;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #6B7280;
            text-align: center;
            margin-bottom: 2rem;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="main-header">🏭 InfiniMetrics 测试结果展示平台</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">AI加速卡通信、算力、推理性能一站式分析与可视化</div>',
        unsafe_allow_html=True,
    )
