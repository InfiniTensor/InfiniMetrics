#!/usr/bin/env python3
"""Common utilities for dashboard pages."""

import streamlit as st
import sys
from pathlib import Path


def init_page(page_title: str, page_icon: str):
    """
    Common page initialization:
    - -Set Streamlit page configuration
    - initialize DataLoader
    - set project path
    """
    # Add project root to Python path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    # Page configuration
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")

    # Initialize use_mongodb setting if not exists
    if "use_mongodb" not in st.session_state:
        st.session_state.use_mongodb = False

    # Initialize DataLoader (respect MongoDB setting)
    if "data_loader" not in st.session_state:
        from utils.data_loader import InfiniMetricsDataLoader

        st.session_state.data_loader = InfiniMetricsDataLoader(
            use_mongodb=st.session_state.use_mongodb,
            fallback_to_files=True,
        )
