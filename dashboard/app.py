#!/usr/bin/env python3
"""Main Streamlit application for InfiniBench dashboard."""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys
from datetime import datetime
from typing import Optional
import traceback
import base64

from infinibench.common.constants import AcceleratorType

project_root = Path(__file__).parent
sys.path.append(str(project_root))

from components.header import render_header
from utils.data_loader import InfiniBenchDataLoader
from common import show_data_source_info

st.set_page_config(
    page_title="InfiniBench Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 全局样式 ──────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* 左侧导航 */
[data-testid="stSidebarNav"] a span { font-size:1.35em !important; font-weight:500; }
section[data-testid="stSidebar"]    { font-size:1.2em; }
.main .block-container              { font-size:1.2em; max-width:1400px; }
[data-testid="stTabs"] button       { font-size:1.15em !important; font-weight:600; }

/* ── 卡片按钮完全透明且覆盖卡片 ── */
div[data-testid="stVerticalBlock"] div[data-testid="stButton"] button {
    position: absolute;
    top: -180px;          /* 向上覆盖卡片高度 */
    left: 0;
    right: 0;
    height: 190px;        /* 覆盖整个卡片区域 */
    opacity: 0 !important;
    cursor: pointer !important;
    z-index: 999;
    border: none !important;
    background: transparent !important;
}
/* 每列相对定位，让按钮绝对定位生效 */
div[data-testid="stVerticalBlock"] {
    position: relative;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state ─────────────────────────────────────────────────────────────
if "data_loader" not in st.session_state:
    st.session_state.data_loader = InfiniBenchDataLoader()
if "selected_platform_keys" not in st.session_state:
    st.session_state.selected_platform_keys = ["nvidia", "mthreads", "cambricon"]
if "use_mongodb" not in st.session_state:
    st.session_state.use_mongodb = False

STATIC_DIR = Path(__file__).parent / "static" / "logos"

PLATFORMS = [
    {"key": "nvidia", "label": "NVIDIA A100", "vendor": "NVIDIA", "logo": "nvidia.png"},
    {
        "key": "mthreads",
        "label": "摩尔线程",
        "vendor": "Moore Threads",
        "logo": "mthreads.png",
    },
    {
        "key": "cambricon",
        "label": "寒武纪",
        "vendor": "Cambricon",
        "logo": "cambricon.png",
    },
    {"key": "metax", "label": "沐曦", "vendor": "MetaX", "logo": "metax.png"},
    {"key": "iluvatar", "label": "天数智芯", "vendor": "Iluvatar", "logo": "iluvatar.png"},
    {"key": "ascend", "label": "昇腾", "vendor": "Huawei", "logo": "ascend.png"},
    {"key": "hygon", "label": "海光", "vendor": "Hygon", "logo": "hygon.png"},
    {"key": "generic", "label": "阿里 PPU", "vendor": "Alibaba", "logo": "ali.png"},
]

PLATFORM_LABEL_MAP = {p["key"]: p["label"] for p in PLATFORMS}

PLATFORM_COLORS = {
    "nvidia": "#76b900",
    "mthreads": "#0066cc",
    "cambricon": "#c0392b",
    "metax": "#1a73e8",
    "iluvatar": "#e65c00",
    "ascend": "#cf0a2c",
    "hygon": "#c8a800",
    "generic": "#ff6a00",
}

# run_id / path 中推断平台用的别名（小写）
ACC_ALIASES = {
    "nvidia": ["nv_a100", "nvidia"],
    "mthreads": ["mthreads", "moore"],
    "cambricon": ["cambricon", "mlu"],
    "metax": ["metax", "mx"],
    "iluvatar": ["iluvatar", "iluvatarcore"],
    "ascend": ["ascend", "npu"],
    "hygon": ["hygon", "dcu"],
    "generic": ["ali_ppu", "generic", "alibaba", "ppu"],
}

DIMENSIONS = [
    {"key": "infer", "label": "🚀 推理", "unit": "tokens/s", "metric_kw": "throughput"},
    {"key": "comm", "label": "🔗 通信", "unit": "GB/s", "metric_kw": "bandwidth"},
    {"key": "train", "label": "🏋️ 训练", "unit": "samples/s", "metric_kw": "throughput"},
    {"key": "operator", "label": "⚡ 算子", "unit": "TFLOPS", "metric_kw": "flops"},
    {"key": "hardware", "label": "🔧 硬件", "unit": "GB/s", "metric_kw": "bandwidth"},
]


def _infer_acc_from_run_id(run_id: str) -> Optional[str]:
    """从 run_id 或 path 字符串中推断平台 key。"""
    rid_lower = (run_id or "").lower()
    for acc_key, aliases in ACC_ALIASES.items():
        if any(a in rid_lower for a in aliases):
            return acc_key
    return None


def _logo_img_tag(logo_file: str, size_px: int = 96) -> str:
    path = STATIC_DIR / logo_file
    key = logo_file.rsplit(".", 1)[0]
    color = PLATFORM_COLORS.get(key, "#888")
    if path.exists():
        try:
            raw = path.read_bytes()
            mime = "image/png" if path.suffix == ".png" else "image/jpeg"
            b64 = base64.b64encode(raw).decode()
            return (
                f'<img src="data:{mime};base64,{b64}" '
                f'style="height:{size_px}px; max-width:90%; '
                f"object-fit:contain; display:block; "
                f'margin:0 auto 8px auto;"/>'
            )
        except Exception:
            pass
    letter = key[0].upper() if key else "?"
    sz = int(size_px * 0.45)
    return (
        f'<div style="width:{size_px}px;height:{size_px}px;border-radius:50%;'
        f"background:{color}22;border:2.5px solid {color};"
        f"display:flex;align-items:center;justify-content:center;"
        f"margin:0 auto 8px auto;font-size:{sz}px;"
        f'font-weight:700;color:{color};">{letter}</div>'
    )


def parse_timestamp(ts) -> Optional[datetime]:
    try:
        ts_str = str(ts)
        if "_" in ts_str and len(ts_str) == 15:
            return datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except Exception:
        return None


def format_time(ts) -> str:
    dt = parse_timestamp(ts)
    if dt:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return str(ts)[:19] if ts else "未知"


def _fmt_val(peak: float) -> str:
    if peak >= 1_000_000:
        return f"{peak / 1_000_000:.2f}M"
    if peak >= 1000:
        return f"{peak / 1000:.2f}K"
    return f"{peak:.1f}"


# ── 卡片 HTML 构建（纯字符串，交给 components.html 渲染）────────────────────


def _build_cards_html(flat: list, max_val: float, unit: str) -> str:
    """
    flat: [(acc, model, {peak, config_info}), ...]  已按 peak 降序
    返回完整 HTML 字符串，由 components.v1.html 在独立 iframe 中渲染。
    这样彻底绕开 st.markdown 在 st.tabs 内的 HTML 解析问题。
    """
    N_COL = 4

    style = """
<style>
* { box-sizing: border-box; margin: 0; padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif; }
body { background: transparent; padding: 4px 0; }
.row { display: flex; gap: 16px; margin-bottom: 16px; }
.card {
  flex: 1; min-width: 0;
  border-radius: 14px; overflow: hidden;
  background: #fff; border: 1.5px solid #e8e8e8;
}
.card-bar  { height: 5px; }
.card-body { padding: 16px 16px 14px 16px; }
.card-platform { font-size: 1.12em; font-weight: 700; color: #222; margin-bottom: 3px; }
.card-model    { font-size: 0.88em; color: #999; margin-bottom: 12px; }
.card-val-row  { margin-bottom: 8px; line-height: 1; }
.card-val      { font-size: 3.0em; font-weight: 900; }
.card-unit     { font-size: 0.88em; color: #aaa; margin-left: 6px; vertical-align: middle; }
.card-cfg      { font-size: 0.82em; color: #bbb; margin-bottom: 12px; }
.prog-bg       { background: #f0f0f0; border-radius: 6px; height: 7px; overflow: hidden; }
.prog-bar      { height: 100%; border-radius: 6px; }
.prog-pct      { font-size: 0.78em; color: #ccc; text-align: right; margin-top: 3px; }
.empty {
  text-align: center; padding: 60px 20px; color: #ccc;
  font-size: 1.1em; border: 2px dashed #e8e8e8;
  border-radius: 14px; margin: 12px 0; line-height: 2;
}
.placeholder { flex: 1; }
</style>
"""

    if not flat:
        return (
            style
            + '<div class="empty">暂无该维度数据<br>'
            + '<span style="font-size:0.85em;">完成相关平台测试后结果将自动展示</span></div>'
        )

    rows_html = []
    for i in range(0, len(flat), N_COL):
        chunk = flat[i : i + N_COL]
        cards = []
        for acc, model, info in chunk:
            color = PLATFORM_COLORS.get(acc, "#888")
            label = PLATFORM_LABEL_MAP.get(acc, acc)
            peak = info["peak"]
            cfg_s = info["config_info"]
            pct = peak / max_val
            val_str = _fmt_val(peak)
            bar_w = int(pct * 100)
            shadow = f"0 2px 10px {color}20"

            cards.append(
                f'<div class="card" style="border-color:{color}44; box-shadow:{shadow};">'
                f'  <div class="card-bar" style="background:{color};"></div>'
                f'  <div class="card-body">'
                f'    <div class="card-platform">{label}</div>'
                f'    <div class="card-model">{model}</div>'
                f'    <div class="card-val-row">'
                f'      <span class="card-val" style="color:{color};">{val_str}</span>'
                f'      <span class="card-unit">{unit}</span>'
                f"    </div>"
                f'    <div class="card-cfg">{cfg_s}</div>'
                f'    <div class="prog-bg">'
                f'      <div class="prog-bar" style="width:{bar_w}%; background:{color};"></div>'
                f"    </div>"
                f'    <div class="prog-pct">{int(pct * 100)}% of best</div>'
                f"  </div>"
                f"</div>"
            )

        # 补空占位保持行对齐
        for _ in range(N_COL - len(chunk)):
            cards.append('<div class="placeholder"></div>')

        rows_html.append('<div class="row">' + "".join(cards) + "</div>")

    return style + "\n".join(rows_html)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    with st.sidebar:
        st.markdown("## ⚙️ 设置")
        use_mongodb = st.toggle("使用 MongoDB", value=st.session_state.use_mongodb)
        if use_mongodb != st.session_state.use_mongodb:
            st.session_state.use_mongodb = use_mongodb
            st.session_state.data_loader = (
                InfiniBenchDataLoader(use_mongodb=True, fallback_to_files=True)
                if use_mongodb
                else InfiniBenchDataLoader()
            )
        show_data_source_info(style="sidebar")
        st.markdown("---")
        results_dir = st.text_input("测试结果目录", value="./output")
        if not use_mongodb and results_dir != str(
            st.session_state.data_loader.results_dir
        ):
            st.session_state.data_loader = InfiniBenchDataLoader(results_dir)
        auto_refresh = st.toggle("自动刷新", value=False)
        if auto_refresh:
            st.rerun()
        st.markdown("---")
        run_id_filter = st.text_input("🔍 Run ID 模糊搜索")

    render_dashboard(run_id_filter)


def render_dashboard(run_id_filter: str = ""):
    # 读取自定义 logo
    _logo_path = Path(__file__).parent / "static" / "logos" / "logo.png"
    _logo_html = ""
    if _logo_path.exists():
        try:
            _b64 = base64.b64encode(_logo_path.read_bytes()).decode()
            _logo_html = f'<img src="data:image/png;base64,{_b64}" style="height:72px;vertical-align:middle;margin-right:16px;"/>'
        except Exception:
            pass

    st.markdown(
        f"""
        <div style="text-align:center;padding:40px 0 16px 0;">
            <div style="font-size:3.2em;font-weight:800;letter-spacing:0.01em;display:flex;align-items:center;justify-content:center;">
                {_logo_html}InfiniBench 结果展示平台
            </div>
            <div style="font-size:1.4em;color:#555;margin-top:14px;">
                AI 加速卡通信 · 算力 · 推理性能一站式分析与可视化
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    cols = st.columns(len(PLATFORMS))
    for idx, p in enumerate(PLATFORMS):
        with cols[idx]:
            is_sel = p["key"] in st.session_state.selected_platform_keys
            border_color = (
                PLATFORM_COLORS.get(p["key"], "#aaa") if is_sel else "#e0e0e0"
            )
            bg_color = "#f0f7ff" if is_sel else "#fafafa"
            shadow = f"0 3px 14px {border_color}55" if is_sel else "none"
            status_html = (
                f'<div style="margin-top:10px;font-size:1.1em;font-weight:700;'
                f'color:{PLATFORM_COLORS.get(p["key"],"#333")};">✔ 已选</div>'
                if is_sel
                else '<div style="margin-top:10px;font-size:1em;color:#ccc;">— 未选 —</div>'
            )
            logo_html = _logo_img_tag(p["logo"], size_px=96)

            st.markdown(
                f"""
                <div style="
                    border:2.5px solid {border_color};
                    border-radius:14px;
                    padding:20px 8px 14px 8px;
                    text-align:center;
                    background:{bg_color};
                    min-height:180px;
                    box-shadow:{shadow};
                    cursor:pointer;
                ">
                    {logo_html}
                    <div style="font-size:1.15em;font-weight:700;color:#222;">{p['label']}</div>
                    {status_html}
                </div>
            """,
                unsafe_allow_html=True,
            )

            # 透明按钮覆盖卡片
            if st.button(" ", key=f"plat_{p['key']}", use_container_width=True):
                keys = list(st.session_state.selected_platform_keys)
                if p["key"] in keys:
                    keys.remove(p["key"])
                else:
                    keys.append(p["key"])
                st.session_state.selected_platform_keys = keys
                st.rerun()

    selected_accs = st.session_state.selected_platform_keys
    st.session_state.selected_accelerators = selected_accs
    st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
    st.divider()

    # ── 数据 ─────────────────────────────────────────────────────────────────
    try:
        all_runs = st.session_state.data_loader.list_test_runs()
        ci_summaries = st.session_state.data_loader.load_summaries()

        def _run_matches(r):
            acc_list = r.get("accelerator_types", [])
            if acc_list:
                return bool(set(acc_list) & set(selected_accs))
            inferred = _infer_acc_from_run_id(r.get("run_id", "") or r.get("path", ""))
            return inferred in selected_accs

        runs = [r for r in all_runs if _run_matches(r)] if selected_accs else all_runs

        if run_id_filter:
            runs = [r for r in runs if run_id_filter in r.get("run_id", "")]

        def _matches_dim(r, prefix):
            tc = (r.get("testcase", "") or "").lower()
            rid = (r.get("run_id", "") or "").lower()
            return tc.startswith(prefix) or prefix in rid

        infer_runs = [r for r in runs if _matches_dim(r, "infer")]
        comm_runs = [r for r in runs if _matches_dim(r, "comm")]
        train_runs = [r for r in runs if _matches_dim(r, "train")]
        ops_runs = [r for r in runs if _matches_dim(r, "operator")]
        hw_runs = [r for r in runs if _matches_dim(r, "hardware")]

        # ── 概览：总测试数 + 成功率 ───────────────────────────────────────────
        st.markdown(
            """
            <div style="font-size:2em;font-weight:700;margin-bottom:20px;">
                📊 当前平台测试概览
            </div>
        """,
            unsafe_allow_html=True,
        )

        total = len(runs)
        success = sum(1 for r in runs if r.get("success"))
        rate = f"{success/total*100:.1f}%" if total > 0 else "—"
        rate_color = "#27ae60" if (total > 0 and success == total) else "#e67e22"

        col_l, col_r = st.columns(2)
        col_l.markdown(
            f"""
            <div style="text-align:center;padding:52px 20px;
                        border:2px solid #0066cc44;border-radius:18px;background:#f0f7ff;">
                <div style="font-size:5em;font-weight:900;color:#0066cc;line-height:1;">{total}</div>
                <div style="font-size:1.5em;color:#444;margin-top:16px;font-weight:600;">总测试数</div>
            </div>
        """,
            unsafe_allow_html=True,
        )
        col_r.markdown(
            f"""
            <div style="text-align:center;padding:52px 20px;
                        border:2px solid {rate_color}44;border-radius:18px;background:#f0fff4;">
                <div style="font-size:5em;font-weight:900;color:{rate_color};line-height:1;">{rate}</div>
                <div style="font-size:1.5em;color:#444;margin-top:16px;font-weight:600;">成功率</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)
        st.divider()

        # ── 多维度跨平台对比 Tab ──────────────────────────────────────────────
        st.markdown(
            """
            <div style="font-size:2em;font-weight:700;margin-bottom:6px;">
                📈 多维度跨平台性能对比
            </div>
            <div style="font-size:1.1em;color:#666;margin-bottom:16px;">
                选择上方平台卡片，对比各维度性能峰值
            </div>
        """,
            unsafe_allow_html=True,
        )

        dim_runs_map = {
            "infer": infer_runs,
            "comm": comm_runs,
            "train": train_runs,
            "operator": ops_runs,
            "hardware": hw_runs,
        }

        tabs = st.tabs([d["label"] for d in DIMENSIONS])
        for tab, dim in zip(tabs, DIMENSIONS):
            with tab:
                render_platform_cards(
                    dim_runs_map.get(dim["key"], []),
                    selected_accs,
                    unit=dim["unit"],
                    metric_kw=dim["metric_kw"],
                )

        st.divider()

        # ── 折叠区 ───────────────────────────────────────────────────────────
        with st.expander("📈 CI 运行统计", expanded=False):
            render_ci_stats(ci_summaries)
        with st.expander("🧾 Dispatcher 汇总记录", expanded=False):
            render_dispatcher_summary(ci_summaries)
        with st.expander("📋 CI 详细记录", expanded=False):
            render_ci_detailed_table(ci_summaries)
        with st.expander("🔍 失败详情", expanded=False):
            render_failure_details(ci_summaries)

    except Exception as e:
        st.error(f"Dashboard 加载失败: {e}")
        st.code(traceback.format_exc())


# ── 多维度卡片展示（使用 components.html）───────────────────────────────────
def render_platform_cards(dim_runs, selected_accs, unit, metric_kw):
    """
    使用 st.components.v1.html 渲染卡片，彻底绕开
    st.markdown 在 st.tabs 内的 HTML 解析 bug。
    """
    platform_results: dict = {}

    for r in dim_runs:
        acc_list = list(r.get("accelerator_types", []) or [])
        if not acc_list:
            rid = r.get("run_id", "") or r.get("path", "") or ""
            inferred = _infer_acc_from_run_id(rid)
            if inferred:
                acc_list = [inferred]
        if not acc_list:
            continue
        acc = acc_list[0]
        if acc not in selected_accs:
            continue

        # 模型名（优先 config，其次解析 run_id）
        model = (
            r.get("config", {}).get("model")
            or r.get("config", {}).get("model_name")
            or ""
        )
        if not model:
            rid = r.get("run_id", "") or ""
            parts = rid.split(".")
            model = parts[3] if len(parts) >= 4 else (rid or "未知模型")

        identifier = r.get("path") or r.get("run_id")
        try:
            data = st.session_state.data_loader.load_test_result(identifier)
            cfg = data.get("config", {})
            ia = cfg.get("infer_args", {})
            bs = ia.get("static_batch_size", ia.get("batch_size", "—"))
            pt = ia.get("prompt_token_num", ia.get("input_len", "—"))
            ot = ia.get("output_token_num", ia.get("output_len", "—"))
            config_info = f"batch={bs} | in={pt} | out={ot}"

            for m in data.get("metrics", []):
                if m.get("data") is None:
                    continue
                if metric_kw not in m.get("name", ""):
                    continue
                df = m["data"]
                ycol = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                try:
                    peak = float(df[ycol].max())
                except Exception:
                    continue

                platform_results.setdefault(acc, {})
                prev = platform_results[acc].get(model)
                if prev is None or peak > prev["peak"]:
                    platform_results[acc][model] = {
                        "peak": peak,
                        "config_info": config_info,
                    }
        except Exception:
            continue

    # 展开并按峰值降序
    flat = [
        (acc, model, info)
        for acc, models in platform_results.items()
        for model, info in models.items()
    ]
    flat.sort(key=lambda x: x[2]["peak"], reverse=True)

    max_val = max(x[2]["peak"] for x in flat) if flat else 1.0

    # 动态计算 iframe 高度
    n_rows = max(1, -(-len(flat) // 4))  # ceiling division by 4
    height = n_rows * 215 + 20 if flat else 140

    html = _build_cards_html(flat, max_val, unit)
    components.html(html, height=height, scrolling=False)


# ── CI 统计 ───────────────────────────────────────────────────────────────────
def render_ci_stats(ci_summaries):
    if not ci_summaries:
        st.info("暂无 CI 汇总数据")
        return
    total_cases = sum(s.get("total_tests", 0) for s in ci_summaries)
    total_success = sum(s.get("successful_tests", 0) for s in ci_summaries)
    total_failed = sum(s.get("failed_tests", 0) for s in ci_summaries)
    total_runs = len(ci_summaries)
    avg_rate = (total_success / total_cases * 100) if total_cases > 0 else 0
    recent = ci_summaries[:10]
    rec_s = sum(s.get("successful_tests", 0) for s in recent)
    rec_t = sum(s.get("total_tests", 0) for s in recent)
    rec_rate = (rec_s / rec_t * 100) if rec_t > 0 else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("CI运行次数", total_runs)
    c2.metric("测试用例总数", f"{total_cases:,}")
    c3.metric("通过用例", f"{total_success:,}")
    c4.metric("失败用例", f"{total_failed:,}")
    c5.metric("平均成功率", f"{avg_rate:.1f}%")
    c6.metric("最近10次", f"{rec_rate:.1f}%", delta=f"{rec_rate-avg_rate:.1f}%")

    daily = {}
    for s in ci_summaries:
        dt = parse_timestamp(s.get("timestamp", ""))
        if not dt:
            continue
        dk = dt.strftime("%Y-%m-%d")
        daily.setdefault(dk, {"total": 0, "success": 0})
        daily[dk]["total"] += s.get("total_tests", 0)
        daily[dk]["success"] += s.get("successful_tests", 0)

    if daily:
        dates = sorted(daily.keys())
        totals = [daily[d]["total"] for d in dates]
        successes = [daily[d]["success"] for d in dates]
        failures = [t - s for t, s in zip(totals, successes)]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=dates, y=totals, name="总测试数", marker_color="lightblue"))
        fig.add_trace(
            go.Bar(x=dates, y=successes, name="成功", marker_color="lightgreen")
        )
        fig.add_trace(go.Bar(x=dates, y=failures, name="失败", marker_color="lightcoral"))
        fig.update_layout(
            barmode="group",
            template="plotly_white",
            height=360,
            xaxis_tickangle=-45,
            font=dict(size=13),
        )
        st.plotly_chart(fig, use_container_width=True)


def render_dispatcher_summary(ci_summaries):
    if not ci_summaries:
        return
    rows = []
    for s in ci_summaries[:15]:
        total = s.get("total_tests", 0)
        rows.append(
            {
                "时间": format_time(s.get("timestamp", "")),
                "总测试数": total,
                "成功": s.get("successful_tests", 0),
                "失败": s.get("failed_tests", 0),
                "成功率": f"{s.get('successful_tests',0)/total*100:.1f}%"
                if total > 0
                else "-",
                "文件": s.get("file", ""),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_ci_detailed_table(ci_summaries):
    if not ci_summaries:
        st.info("未找到 CI 汇总记录")
        return
    rows = []
    for s in ci_summaries[:30]:
        total = s.get("total_tests", 0)
        success = s.get("successful_tests", 0)
        failed = s.get("failed_tests", 0)
        duration = s.get("duration", s.get("total_duration_seconds", 0))
        results = s.get("results", [])
        git = s.get("git", {})
        sc = git.get("short_commit", "")
        cm = git.get("commit_message", "")
        branch = git.get("branch", "")
        rows.append(
            {
                "时间": format_time(s.get("timestamp", "")),
                "Run ID": results[0].get("run_id", "-") if results else "-",
                "Commit": sc[:8] if sc and sc != "unknown" else "本地",
                "提交信息": (cm[:50] + "...") if len(cm) > 50 else cm or "-",
                "分支": branch
                if branch and not branch.startswith("unknown")
                else "local",
                "作者": git.get("commit_author", "-"),
                "总数": total,
                "✅": success,
                "❌": failed,
                "成功率": f"{success/total*100:.1f}%" if total > 0 else "-",
                "状态": "✅ 成功" if failed == 0 else "❌ 失败",
                "时长": f"{duration:.1f}s" if duration > 0 else "-",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_failure_details(ci_summaries):
    failed_records = []
    for s in ci_summaries:
        if s.get("failed_tests", 0) == 0:
            continue
        git = s.get("git", {})
        sc = git.get("short_commit", "")
        failed_details = s.get("failed_tests_details", []) or [
            {
                "testcase": r.get("testcase", "unknown"),
                "run_id": r.get("run_id", "unknown"),
                "result_code": r.get("result_code", -1),
                "result_file": r.get("result_file", ""),
                "error_msg": r.get("error_msg", ""),
            }
            for r in s.get("results", [])
            if r.get("result_code", 0) != 0
        ]
        if failed_details:
            failed_records.append(
                {
                    "时间": format_time(s.get("timestamp", "")),
                    "Commit": sc[:8] if sc and sc != "unknown" else "本地",
                    "失败数": len(failed_details),
                    "失败详情": failed_details,
                }
            )
    if not failed_records:
        st.info("暂无失败记录")
        return
    for record in failed_records[:15]:
        with st.expander(
            f"📅 {record['时间']} — 失败 {record['失败数']} 个 (Commit: {record['Commit']})"
        ):
            for i, fail in enumerate(record["失败详情"][:20]):
                st.markdown(f"**{i+1}. {fail.get('testcase','unknown')}**")
                st.markdown(f"- Run ID: `{fail.get('run_id','unknown')}`")
                st.markdown(f"- Result Code: {fail.get('result_code',-1)}")
                if fail.get("result_file"):
                    st.markdown(f"- File: `{fail.get('result_file')}`")
                st.divider()
            if len(record["失败详情"]) > 20:
                st.info(f"还有 {len(record['失败详情'])-20} 个未显示")


if __name__ == "__main__":
    main()
