#!/usr/bin/env python3
"""Common utilities for dashboard pages."""

import streamlit as st
import sys
from pathlib import Path


def init_page(page_title: str, page_icon: str):
    """
    通用页面初始化：
    - 设置 Streamlit 页面配置
    - 初始化 DataLoader
    - 设置项目路径
    """

    # Add project root to Python path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    # Page configuration
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")

    # Initialize DataLoader
    if "data_loader" not in st.session_state:
        from utils.data_loader import InfiniMetricsDataLoader

        st.session_state.data_loader = InfiniMetricsDataLoader()
