#!/usr/bin/env python3
"""InfiniBench Dashboard - 国产算力性能基准平台"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys
from datetime import datetime
from typing import Optional, Dict
import traceback
import base64
import json

project_root = Path(__file__).parent
sys.path.append(str(project_root))

# ── Data loading from /data/ directory ───────────────────────────────────────
# No static_data.py needed. Drop CSVs/XLSXs into data/ subdirs and restart.


@st.cache_data(ttl=300)  # reload at most every 5 min; or restart to force reload
def load_all_data():
    import re, pandas as pd
    from pathlib import Path

    DATA_ROOT = Path(__file__).parent / "data"
    OP_NORM = {
        "causal_softmax": "causalsoftmax",
        "rms_norm": "rmsnorm",
        "swiglu": "silu",
        "gemm": "matmul",
        "rmsnorm": "rmsnorm",
        "causalsoftmax": "causalsoftmax",
        "matmul": "matmul",
        "silu": "silu",
        "add": "add",
        "topk": "topk",
        "embedding": "embedding",
        "cast": "cast",
        "cat": "cat",
        "linear": "linear",
        "mul": "mul",
    }
    PLAT_MAP = {
        "ali": "generic",
        "mthreads": "mthreads",
        "metax": "metax",
        "hygon": "hygon",
        "cambricon": "cambricon",
        "nvidia": "nvidia",
        "iluvatar": "iluvatar",
        "ascend": "ascend",
        "moore": "mthreads",
    }

    def safe_f(v):
        try:
            return float(v) if str(v) not in ("nan", "NA", "None", "") else None
        except:
            return None

    # ── Operator data ──────────────────────────────────────────────────────
    ops_raw = {}  # {platform_key: {op_name: [rows]}}
    op_dir = DATA_ROOT / "operator"
    if op_dir.exists():
        for csv_path in sorted(op_dir.glob("*.csv")):
            try:
                df = pd.read_csv(csv_path)
                # Derive platform key from filename prefix (e.g. "hygon_operator_*.csv" → "hygon")
                # This is the fallback when the CSV's platform column is wrong/missing
                fname_stem = csv_path.stem.lower()  # e.g. "iluvatar_operator_20260430"
                fname_plat = fname_stem.split("_")[0]  # e.g. "iluvatar"
                fname_pk = PLAT_MAP.get(fname_plat, fname_plat)
                for _, r in df.iterrows():
                    # Prefer filename-derived key; only use CSV column if filename gives 'ops'/'data' etc.
                    csv_pk = PLAT_MAP.get(
                        str(r.get("platform", "")).strip().lower(),
                        str(r.get("platform", "")).strip().lower(),
                    )
                    # Use filename key if it resolves to a known platform; else use CSV column
                    pk = fname_pk if fname_pk in PLAT_MAP.values() else csv_pk
                    op = OP_NORM.get(
                        str(r["op_name"]).strip().lower(),
                        str(r["op_name"]).strip().lower(),
                    )
                    row = {
                        "shape": str(r["shape_config"]),
                        "dtype": str(r["dtype"]),
                        "ic_lat": safe_f(r.get("ic_latency_ms")),
                        "pt_lat": safe_f(r.get("pt_latency_ms")),
                        "ic_tflops": safe_f(r.get("ic_tflops")),
                        "pt_tflops": safe_f(r.get("pt_tflops")),
                    }
                    ops_raw.setdefault(pk, {}).setdefault(op, []).append(row)
            except Exception as e:
                st.warning(f"operator CSV load error {csv_path.name}: {e}")

    def ops_summary(rows):
        scored = [
            (r["ic_lat"], r["pt_lat"])
            for r in rows
            if r["ic_lat"] and r["pt_lat"] and r["ic_lat"] > 0
        ]
        per_row = [pt / ic * 100 for ic, pt in scored]
        avg_sc = round(sum(per_row) / len(per_row)) if per_row else None
        ic_avg = sum(ic for ic, _ in scored) / len(scored) if scored else None
        pt_avg = sum(pt for _, pt in scored) / len(scored) if scored else None
        best_row, best_sc = None, -1
        for r in rows:
            if r["ic_lat"] and r["pt_lat"] and r["ic_lat"] > 0:
                s = r["pt_lat"] / r["ic_lat"] * 100
                if s > best_sc:
                    best_sc = s
                    best_row = r
        return {
            "avg_score": avg_sc,
            "ic_avg": ic_avg,
            "pt_avg": pt_avg,
            "best_row": best_row,
            "best_row_score": round(best_sc) if best_sc > 0 else None,
            "n": len(rows),
            "rows": rows,
        }

    ops_db = {
        pk: {op: ops_summary(rows) for op, rows in ops.items()}
        for pk, ops in ops_raw.items()
    }

    # ── Inference data ─────────────────────────────────────────────────────
    infer_rows = []
    infer_dir = DATA_ROOT / "infer"
    if infer_dir.exists():
        for fpath in sorted(infer_dir.iterdir()):
            try:
                if fpath.suffix in (".xlsx", ".xls"):
                    # Try header=0 first; fall back to header=10 for old template format
                    df = pd.read_excel(fpath, header=0)
                    df.columns = [str(c).strip() for c in df.columns]
                    if "platform" not in [c.lower() for c in df.columns]:
                        df = pd.read_excel(fpath, header=10)
                        df.columns = [str(c).strip() for c in df.columns]
                elif fpath.suffix == ".csv":
                    # Try utf-8 first, fall back to gbk (common for Excel-exported CSVs)
                    for enc in ["utf-8", "utf-8-sig", "gbk", "gb18030"]:
                        try:
                            df = pd.read_csv(fpath, encoding=enc)
                            break
                        except (UnicodeDecodeError, Exception):
                            continue
                    else:
                        df = pd.read_csv(fpath, encoding="gbk", errors="replace")
                else:
                    continue
                df = df.dropna(subset=["platform"])
                df.columns = [str(c).strip() for c in df.columns]
                df = df.rename(
                    columns={
                        "Prefill吞吐量（tokens/s）": "prefill_tps",
                        "Decode吞吐量（tokens/s）": "decode_tps",
                    }
                )
                for _, r in df.iterrows():
                    pk = PLAT_MAP.get(
                        str(r["platform"]).strip().lower(),
                        str(r["platform"]).strip().lower(),
                    )
                    infer_rows.append(
                        {
                            "platform": pk,
                            "model": str(r.get("model", "")).upper(),
                            "batch": safe_f(r.get("batch_size")),
                            "in_tok": safe_f(r.get("input_tokens")),
                            "out_tok": safe_f(r.get("output_tokens")),
                            "n_gpu": safe_f(r.get("n_gpu")),
                            "dtype": str(r.get("dtype", "")),
                            "il_ttft": safe_f(r.get("il_ttft_ms")),
                            "il_dec": safe_f(r.get("il_decode_ms")),
                            "prefill_tps": safe_f(r.get("prefill_tps")),
                            "decode_tps": safe_f(r.get("decode_tps")),
                            "vl_ttft": safe_f(r.get("vl_ttft_ms")),
                            "vl_dec": safe_f(r.get("vl_decode_ms")),
                            "vl_tps": safe_f(r.get("vl_throughput_tps")),
                        }
                    )
            except Exception as e:
                st.warning(f"infer file load error {fpath.name}: {e}")

    # ── Training data ──────────────────────────────────────────────────────
    train_rows = []
    train_dir = DATA_ROOT / "train"
    if train_dir.exists():
        for fpath in sorted(train_dir.iterdir()):
            try:
                df = (
                    pd.read_excel(fpath, header=None)
                    if fpath.suffix in (".xlsx", ".xls")
                    else pd.read_csv(fpath)
                )
                # find training rows: have 'platform' and 'throughput_tpps' columns
                # Try header=0 first
                for hdr in [0, None]:
                    try:
                        df2 = (
                            pd.read_excel(fpath, header=hdr)
                            if fpath.suffix in (".xlsx", ".xls")
                            else pd.read_csv(fpath, header=hdr)
                        )
                        if "platform" in [str(c).lower() for c in df2.columns]:
                            df2.columns = [str(c).strip() for c in df2.columns]
                            df2 = df2.dropna(subset=["platform"])
                            for _, r in df2.iterrows():
                                if safe_f(r.get("throughput_tpps")) is None:
                                    continue
                                pk = PLAT_MAP.get(
                                    str(r["platform"]).strip().lower(),
                                    str(r["platform"]).strip().lower(),
                                )
                                train_rows.append(
                                    {
                                        "platform": pk,
                                        "framework": str(r.get("framework", "")),
                                        "model": str(r.get("model", "")),
                                        "micro_bs": safe_f(r.get("micro_batch_size")),
                                        "seq_len": safe_f(
                                            r.get("seq_len", r.get("seq_length"))
                                        ),
                                        "n_gpu": safe_f(r.get("n_gpu")),
                                        "dtype": str(r.get("dtype", "")),
                                        "zero_stage": str(r.get("zero_stage", "")),
                                        "flash_attn": str(r.get("flash_attn", "")),
                                        "tpps": safe_f(r.get("throughput_tpps")),
                                        "remarks": str(r.get("remarks", "")),
                                    }
                                )
                            break
                    except Exception:
                        continue
            except Exception as e:
                st.warning(f"train file load error {fpath.name}: {e}")

    # ── Comm data ──────────────────────────────────────────────────────────
    comm_rows = []
    comm_dir = DATA_ROOT / "comm"
    if comm_dir.exists():
        for fpath in sorted(comm_dir.iterdir()):
            try:
                # Read with header row (column names)
                if fpath.suffix in (".xlsx", ".xls"):
                    df = pd.read_excel(fpath, header=0)
                else:
                    for enc in ["utf-8", "utf-8-sig", "gbk", "gb18030"]:
                        try:
                            df = pd.read_csv(fpath, encoding=enc, header=0)
                            break
                        except UnicodeDecodeError:
                            continue
                df.columns = [str(c).strip().lower() for c in df.columns]
                df = df.dropna(subset=["platform"])
                for _, r in df.iterrows():
                    pk = PLAT_MAP.get(
                        str(r["platform"]).strip().lower(),
                        str(r["platform"]).strip().lower(),
                    )
                    bw = safe_f(r.get("bw_gbps", r.get("bw_GBps", r.get("bw", None))))
                    if bw is None:
                        continue
                    comm_rows.append(
                        {
                            "platform": pk,
                            "link_type": str(r.get("link_type", "")).strip(),
                            "comm_type": str(r.get("comm_type", ""))
                            .strip()
                            .replace("\n", ""),
                            "n_gpu": safe_f(r.get("n_gpu")),
                            "bw_GBps": bw,
                            "remarks": str(r.get("remarks", "")),
                        }
                    )
            except Exception as e:
                st.warning(f"comm file load error {fpath.name}: {e}")

    return ops_db, infer_rows, train_rows, comm_rows


OPS_DATA, INFER_DATA, TRAIN_DATA, COMM_DATA = load_all_data()

from components.header import render_header
from utils.data_loader import InfiniBenchDataLoader
from common import show_data_source_info

st.set_page_config(
    page_title="InfiniBench 国产算力性能基准平台",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
[data-testid="stSidebarNav"] a span { font-size:1.2em !important; font-weight:500; }
section[data-testid="stSidebar"]    { font-size:1.1em; }
.main .block-container              { max-width:1400px; font-size:1.1em; }
[data-testid="stTabs"] button       { font-size:1.1em !important; font-weight:600; }
div[data-testid="stVerticalBlock"] div[data-testid="stButton"] button {
    position:absolute;top:-180px;left:0;right:0;height:190px;
    opacity:0 !important;cursor:pointer !important;z-index:999;
    border:none !important;background:transparent !important; }
div[data-testid="stVerticalBlock"] { position:relative; }
hr { border-color:#e8e8e8 !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state ──────────────────────────────────────────────────────────────
if "data_loader" not in st.session_state:
    st.session_state.data_loader = InfiniBenchDataLoader()
if "selected_platform_keys" not in st.session_state:
    st.session_state.selected_platform_keys = [
        "nvidia",
        "mthreads",
        "cambricon",
        "ascend",
        "hygon",
        "metax",
        "iluvatar",
        "generic",
    ]
if "use_mongodb" not in st.session_state:
    st.session_state.use_mongodb = False

# ── Static config ──────────────────────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static" / "logos"

PLATFORMS = [
    {"key": "nvidia", "label": "NVIDIA", "sub": "国际标杆", "logo": "nvidia.png"},
    {"key": "mthreads", "label": "摩尔线程", "sub": "国产", "logo": "mthreads.png"},
    {"key": "cambricon", "label": "寒武纪", "sub": "国产", "logo": "cambricon.png"},
    {"key": "metax", "label": "沐曦", "sub": "国产", "logo": "metax.png"},
    {"key": "iluvatar", "label": "天数智芯", "sub": "国产", "logo": "iluvatar.png"},
    {"key": "ascend", "label": "昇腾", "sub": "国产", "logo": "ascend.png"},
    {"key": "hygon", "label": "海光", "sub": "国产", "logo": "hygon.png"},
    {"key": "generic", "label": "阿里 PPU", "sub": "国产", "logo": "ali.png"},
]
PLATFORM_LABEL_MAP = {p["key"]: p["label"] for p in PLATFORMS}
PLATFORM_SUB_MAP = {p["key"]: p["sub"] for p in PLATFORMS}
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


# ── Helpers ────────────────────────────────────────────────────────────────────
def _infer_acc(run_id):
    rid = (run_id or "").lower()
    for k, aliases in ACC_ALIASES.items():
        if any(a in rid for a in aliases):
            return k
    return None


def _logo_img_tag(logo_file, size_px=84):
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
                f'style="height:{size_px}px;max-width:90%;object-fit:contain;display:block;margin:0 auto 8px auto;"/>'
            )
        except:
            pass
    letter = key[0].upper() if key else "?"
    sz = int(size_px * 0.45)
    return (
        f'<div style="width:{size_px}px;height:{size_px}px;border-radius:50%;'
        f"background:{color}22;border:2.5px solid {color};display:flex;align-items:center;"
        f'justify-content:center;margin:0 auto 8px auto;font-size:{sz}px;font-weight:700;color:{color};">{letter}</div>'
    )


def parse_timestamp(ts):
    try:
        s = str(ts)
        if "_" in s and len(s) == 15:
            return datetime.strptime(s, "%Y%m%d_%H%M%S")
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except:
        return None


def format_time(ts):
    dt = parse_timestamp(ts)
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else (str(ts)[:19] if ts else "未知")


# ── Build HTML dashboard ───────────────────────────────────────────────────────
def _build_dashboard_html(selected_accs, colors, label_map, sub_map):
    """
    Unified HTML component with:
    - Operator dim  : full OPS_DATA drill-down (same-row scoring, best-row showcase)
    - Inference dim : InfiniLM vs vLLM cards, drill-down with 3 sub-tabs
    - Training dim  : BMTrain/Megatron vs NVIDIA baseline
    - Comm dim      : bandwidth vs NVIDIA
    """
    plats_json = json.dumps(
        [
            {"key": acc, "name": label_map.get(acc, acc), "type": sub_map.get(acc, "")}
            for acc in selected_accs
        ]
    )
    colors_json = json.dumps(colors)
    ops_json = json.dumps(OPS_DATA)
    infer_json = json.dumps(INFER_DATA)
    train_json = json.dumps(TRAIN_DATA)
    comm_json = json.dumps(COMM_DATA)

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0;
   font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif;}}
body{{background:#fff;padding:14px 4px 24px;color:#1a1a2e;}}

/* ── Dim tabs ── */
.dtabs{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;}}
.dtab{{padding:9px 22px;border-radius:24px;border:2px solid #e0e0e0;
       font-size:30px;cursor:pointer;color:#666;background:#fff;font-weight:600;transition:all .14s;}}
.dtab:hover{{border-color:#1a73e8;color:#1a73e8;}}
.dtab.on{{background:#1a73e8;border-color:#1a73e8;color:#fff;}}

/* ── Filter strip ── */
.fstrip{{display:flex;gap:7px;align-items:center;flex-wrap:wrap;margin-bottom:16px;min-height:32px;}}
.fl{{font-size:28px;color:#555;font-weight:700;}}
.fb{{padding:5px 14px;border-radius:14px;border:2px solid #e0e0e0;font-size:26px;
     cursor:pointer;color:#666;background:#fff;font-weight:600;transition:all .12s;}}
.fb:hover{{border-color:#1a73e8;color:#1a73e8;}}
.fb.on{{border-color:#1a73e8;color:#1a73e8;background:#e8f0fe;}}
.fsep{{color:#ddd;margin:0 5px;font-size:30px;}}

/* ── Cards grid ── */
.cgrid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(460px,1fr));gap:16px;}}

/* ── Card ── */
.pcard{{background:#fff;border:2px solid #eee;border-radius:20px;overflow:hidden;
        cursor:pointer;transition:box-shadow .15s,transform .12s;}}
.pcard:hover{{box-shadow:0 6px 28px rgba(0,0,0,.11);transform:translateY(-2px);}}
.pcard-accent{{height:5px;}}
.pcard-head{{padding:13px 18px 9px;border-bottom:1px solid #f4f4f4;
             display:flex;align-items:center;gap:10px;}}
.pcard-name{{font-size:36px;font-weight:900;color:#1a1a2e;flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.pcard-type{{font-size:22px;font-weight:700;padding:2px 9px;border-radius:8px;}}

/* vs_open two-col */
.pcols{{display:flex;}}
.pcol{{flex:1;padding:16px 16px 14px;display:flex;flex-direction:column;align-items:center;gap:4px;}}
.pcol-div{{width:1px;background:#f0f0f0;margin:10px 0;}}
.fw-badge{{font-size:26px;font-weight:900;padding:4px 12px;border-radius:10px;margin-bottom:3px;}}
.col-score{{font-size:112px;font-weight:900;line-height:1;margin:5px 0 0;letter-spacing:-.03em;}}
.col-score-lbl{{font-size:24px;color:#aaa;margin-bottom:4px;font-weight:600;}}
.col-bar-t{{width:100%;height:7px;background:#f0f0f0;border-radius:4px;overflow:hidden;margin:5px 0;}}
.col-bar-f{{height:100%;border-radius:4px;}}
.col-peak{{font-size:36px;font-weight:900;}}
.col-unit{{font-size:24px;color:#aaa;margin-left:4px;font-weight:600;}}
.col-cfg{{font-size:22px;color:#bbb;text-align:center;line-height:1.5;margin-top:3px;}}
.col-adv-wrap{{text-align:center;padding:6px 16px 12px;}}
.col-adv{{font-size:28px;font-weight:900;padding:5px 14px;border-radius:12px;display:inline-block;}}
.nodata{{font-size:26px;color:#ccc;font-style:italic;margin-top:16px;}}

/* vs_baseline ring */
.psingle{{padding:16px 18px 14px;display:flex;align-items:center;gap:16px;}}
.ring-w{{width:110px;height:110px;flex-shrink:0;position:relative;}}
.ring-w svg{{width:110px;height:110px;transform:rotate(-90deg);}}
.ring-c{{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;line-height:1;}}
.ring-s{{font-size:44px;font-weight:900;}}
.ring-d{{font-size:20px;color:#aaa;display:block;margin-top:2px;font-weight:600;}}
.si{{flex:1;min-width:0;}}
.si-vs{{font-size:24px;color:#aaa;margin-bottom:7px;font-weight:700;}}
.fw-row{{display:flex;align-items:center;gap:7px;margin-top:7px;}}
.fw-row-name{{font-size:26px;font-weight:900;width:70px;flex-shrink:0;}}
.fw-bar-t{{flex:1;height:5px;background:#f0f0f0;border-radius:3px;overflow:hidden;}}
.fw-bar-f{{height:100%;border-radius:3px;}}
.fw-row-s{{font-size:26px;font-weight:900;width:32px;text-align:right;flex-shrink:0;}}
.si-cfg{{font-size:22px;color:#bbb;margin-top:8px;line-height:1.5;}}

.tap-h{{font-size:24px;color:#bbb;text-align:center;padding:8px 0 6px;
        border-top:1px solid #f8f8f8;letter-spacing:.02em;font-weight:600;}}
.empty{{text-align:center;padding:56px 20px;color:#ccc;font-size:30px;
        border:2px dashed #eee;border-radius:16px;grid-column:1/-1;}}

/* ── Modal ── */
.ov-bg{{position:fixed;inset:0;background:rgba(0,0,0,.42);z-index:400;}}
.ov-wrap{{position:fixed;inset:0;display:flex;align-items:center;justify-content:center;z-index:401;padding:16px;}}
.modal{{background:#fff;border-radius:22px;width:880px;max-width:98vw;
        max-height:92vh;overflow:hidden;display:flex;flex-direction:column;
        box-shadow:0 20px 80px rgba(0,0,0,.26);}}
.modal-head{{padding:22px 26px 0;flex-shrink:0;}}
.modal-title-row{{display:flex;align-items:center;gap:10px;margin-bottom:3px;}}
.modal-dot{{width:12px;height:12px;border-radius:50%;flex-shrink:0;}}
.modal-title{{font-size:44px;font-weight:900;}}
.modal-sub{{font-size:26px;color:#888;margin-bottom:14px;font-weight:500;}}
.modal-close{{margin-left:auto;background:none;border:none;font-size:44px;cursor:pointer;color:#bbb;}}
.modal-close:hover{{color:#333;}}
.sum-row{{display:flex;gap:10px;margin-bottom:16px;flex-wrap:wrap;}}
.sum-pill{{background:#f7f8fa;border-radius:14px;padding:12px 18px;text-align:center;flex:1;min-width:100px;}}
.sum-val{{font-size:52px;font-weight:900;line-height:1;}}
.sum-lbl{{font-size:22px;color:#aaa;margin-top:4px;font-weight:600;}}
.modal-tabs{{display:flex;gap:0;border-bottom:2px solid #f0f0f0;padding:0 26px;flex-shrink:0;}}
.mtab{{padding:9px 20px;font-size:28px;font-weight:700;cursor:pointer;color:#aaa;
       border-bottom:3px solid transparent;margin-bottom:-2px;transition:all .12s;}}
.mtab.on{{color:#1a73e8;border-bottom-color:#1a73e8;}}
.modal-body{{flex:1;overflow-y:auto;padding:18px 26px 24px;}}
.dtype-row{{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:14px;align-items:center;}}
.dtype-lbl{{font-size:26px;font-weight:700;color:#666;}}
.dtype-btn{{padding:5px 14px;border-radius:11px;border:1.5px solid #e0e0e0;font-size:24px;
            cursor:pointer;color:#666;background:#fff;font-weight:700;transition:all .12s;}}
.dtype-btn:hover{{border-color:#1a73e8;color:#1a73e8;}}
.dtype-btn.on{{border-color:#1a73e8;color:#1a73e8;background:#e8f0fe;}}
.notice{{font-size:24px;color:#555;background:#f0f7ff;border-radius:10px;
         padding:10px 14px;margin-bottom:16px;line-height:1.7;border-left:4px solid #1a73e8;}}

/* ops table */
.dtbl{{width:100%;border-collapse:collapse;font-size:26px;}}
.dtbl thead th{{font-size:22px;font-weight:800;color:#aaa;padding:6px 10px;
                text-align:left;border-bottom:2px solid #f0f0f0;white-space:nowrap;
                position:sticky;top:0;background:#fff;}}
.dtbl tbody tr:hover td{{background:#f8fbff;}}
.dtbl td{{padding:10px 10px;border-bottom:1px solid #f6f6f6;vertical-align:middle;}}
.shape-cell{{font-size:24px;color:#444;font-family:monospace;font-weight:600;}}
.dtype-pill{{display:inline-block;font-size:22px;font-weight:800;padding:2px 9px;border-radius:7px;}}
.score-pill{{display:inline-block;font-size:26px;font-weight:900;padding:3px 10px;border-radius:9px;}}
.dual-bar{{display:flex;flex-direction:column;gap:4px;min-width:130px;}}
.dual-row{{display:flex;align-items:center;gap:5px;}}
.dual-lbl{{font-size:20px;font-weight:700;width:28px;flex-shrink:0;text-align:right;}}
.dual-track{{flex:1;height:5px;background:#f0f0f0;border-radius:3px;overflow:hidden;}}
.dual-fill{{height:100%;border-radius:3px;}}
.dual-ms{{font-size:20px;font-weight:700;width:52px;text-align:right;flex-shrink:0;}}

/* score tab */
.score-tbl{{width:100%;border-collapse:collapse;font-size:26px;}}
.score-tbl th{{font-size:22px;font-weight:800;color:#aaa;padding:6px 10px;text-align:left;border-bottom:2px solid #f0f0f0;}}
.score-tbl td{{padding:10px 10px;border-bottom:1px solid #f6f6f6;vertical-align:middle;}}
.score-tbl tr:last-child td{{border-bottom:none;}}
.best-row-card{{background:#f0f9f0;border:1.5px solid #76b900;border-radius:12px;
                padding:12px 16px;margin-bottom:16px;font-size:26px;line-height:1.8;}}
.score-explain{{background:#fffbf0;border:1.5px solid #f0e0a0;border-radius:12px;
                padding:12px 16px;margin-bottom:16px;font-size:24px;line-height:1.8;color:#555;}}
.placeholder{{background:#f8f9fa;border-radius:12px;padding:20px;text-align:center;
              color:#aaa;font-size:28px;border:2px dashed #e0e0e0;}}

/* infer tabs */
.itab-row{{display:flex;gap:4px;margin-bottom:14px;}}
.itab{{padding:5px 14px;border-radius:10px;border:1.5px solid #e0e0e0;font-size:24px;
       cursor:pointer;color:#666;background:#fff;font-weight:600;}}
.itab.on{{border-color:#1a73e8;color:#1a73e8;background:#e8f0fe;}}

/* infer detail table */
.itbl{{width:100%;border-collapse:collapse;font-size:26px;margin-top:8px;}}
.itbl th{{font-size:22px;font-weight:800;color:#aaa;padding:6px 8px;text-align:left;
          border-bottom:2px solid #f0f0f0;white-space:nowrap;}}
.itbl td{{padding:8px 8px;border-bottom:1px solid #f6f6f6;vertical-align:middle;}}
.itbl tr:last-child td{{border-bottom:none;}}
.itbl tr:hover td{{background:#f8fbff;}}
.val-good{{color:#2e7d32;font-weight:800;}}
.val-na{{color:#ccc;font-style:italic;}}

/* train/comm table */
.gen-tbl{{width:100%;border-collapse:collapse;font-size:26px;}}
.gen-tbl th{{font-size:22px;font-weight:800;color:#aaa;padding:7px 10px;text-align:left;border-bottom:2px solid #f0f0f0;}}
.gen-tbl td{{padding:10px 10px;border-bottom:1px solid #f6f6f6;vertical-align:middle;}}
.gen-tbl tr:last-child td{{border-bottom:none;}}
.pct-badge{{display:inline-block;font-size:24px;font-weight:700;padding:2px 9px;border-radius:8px;}}
</style>
</head>
<body>

<div class="dtabs" id="dtabs"></div>
<div class="fstrip" id="fstrip"></div>
<div class="cgrid"  id="cgrid"></div>

<script>
const PLATS  = {plats_json};
const COLORS = {colors_json};
const OPS    = {ops_json};
const INFER  = {infer_json};
const TRAIN  = {train_json};
const COMM   = {comm_json};

const DTYPE_COLORS = {{fp16:'#1a73e8',bf16:'#7b61ff',fp32:'#e65c00'}};
const OP_LABELS = {{causalsoftmax:'CausalSoftmax',rmsnorm:'RMSNorm',embedding:'Embedding',
                    topk:'TopK',add:'Add',matmul:'MatMul',silu:'SiLU',
                    cast:'Cast',cat:'Cat',linear:'Linear',mul:'Mul'}};

const DIMS = [
  {{key:'operator', label:'⚡ 算子',   mode:'ops',   unit:'ms',
    filters:[{{key:'op',label:'算子',opts:Object.keys(OP_LABELS)}}]}},
  {{key:'infer',    label:'🚀 推理',   mode:'infer', unit:'tokens/s',
    filters:[{{key:'batch',label:'Batch',   opts:['1','4','16','64']}},
             {{key:'in',   label:'In-len',  opts:['32','256','4096']}}]}},
  {{key:'train',    label:'🏋️ 训练',  mode:'train', unit:'tokens/s',
    filters:[{{key:'fw',label:'框架',opts:['megatron','bmtrain']}}]}},
  {{key:'comm',     label:'🔗 通信',   mode:'comm',  unit:'GB/s',
    filters:[{{key:'ct',label:'类型',opts:['p2p','allreduce']}}]}},
];

let aDim=0, filt={{}}, modalEl=null, modalBg=null;
let mState=null; // {{pk, mode, activeTab, activeDtype, activeOp, activeInferTab}}

/* ── Helpers ── */
function fmtMs(v){{return v==null?"—":v.toFixed(4)+"ms";}}
function fmtV(v,dp=1){{
  if(v==null||v===undefined) return "—";
  if(v>=1000) return (v/1000).toFixed(1)+"K";
  return v.toFixed(dp);
}}
function scColor(s){{
  if(s==null) return "#aaa";
  if(s>=500) return "#0d47a1";
  if(s>=100) return "#2e7d32";
  if(s>=60)  return "#e65100";
  return "#c62828";
}}

/* ── Tabs ── */
function renderTabs(){{
  document.getElementById("dtabs").innerHTML=
    DIMS.map((d,i)=>`<div class="dtab${{i===aDim?" on":""}}" onclick="setDim(${{i}})">${{d.label}}</div>`).join("");
}}

/* ── Filters ── */
function renderFilters(){{
  const dim=DIMS[aDim], el=document.getElementById("fstrip");
  if(!dim.filters||!dim.filters.length){{el.innerHTML="";return;}}
  el.innerHTML=dim.filters.map((f,fi)=>
    (fi>0?'<span class="fsep">|</span>':"")+
    `<span class="fl">${{f.label}}：</span>`+
    `<button class="fb${{!filt[f.key]?" on":""}}" onclick="clrF('${{f.key}}')">全部</button>`+
    f.opts.map(o=>`<button class="fb${{filt[f.key]===o?" on":""}}" onclick="setF('${{f.key}}','${{o}}')">${{o}}</button>`).join("")
  ).join("");
}}
function setF(k,v){{filt[k]=(filt[k]===v?null:v);renderFilters();renderCards();}}
function clrF(k){{filt[k]=null;renderFilters();renderCards();}}

/* ─────────────────────────────────────────────────────────────────────────────
   CARDS
───────────────────────────────────────────────────────────────────────────── */
function renderCards(){{
  const dim=DIMS[aDim], grid=document.getElementById("cgrid");
  if(!PLATS.length){{grid.innerHTML='<div class="empty">请在上方选择平台</div>';return;}}

  grid.innerHTML=PLATS.map(pk=>{{
    const p=PLATS.find(x=>x.key===pk.key)||pk;
    const c=COLORS[p.key]||"#888";

    if(dim.mode==="ops")     return buildOpsCard(p,c,dim);
    if(dim.mode==="infer")   return buildInferCard(p,c,dim);
    if(dim.mode==="train")   return buildTrainCard(p,c);
    if(dim.mode==="comm")    return buildCommCard(p,c);
    return "";
  }}).filter(Boolean).join("");
}}

/* ── OPS card ── */
function buildOpsCard(p,c,dim){{
  const selOp=filt['op']||null;
  const platOps=OPS[p.key]||{{}};
  // Hide card if platform has no ops data at all
  const hasAnyData=Object.values(platOps).some(od=>od&&od.n>0);
  if(!hasAnyData) return '';

  // Compute displayed score:
  // - op selected → best single-row score (max per-row pt/ic ratio) for that op
  // - all ops     → avg of each op's avg_score
  let dispScore=null, nRows=0, icAvg=null, ptAvg=null;
  if(selOp){{
    const od=platOps[selOp];
    if(od){{
      dispScore=od.best_row_score;   // ← max single-row score (same shape+dtype)
      nRows=od.n; icAvg=od.ic_avg; ptAvg=od.pt_avg;
    }}
  }}else{{
    const scores=[]; let totalN=0;
    Object.values(platOps).forEach(od=>{{if(od.avg_score!=null){{scores.push(od.avg_score);totalN+=od.n;}}}});
    if(scores.length){{dispScore=Math.round(scores.reduce((a,b)=>a+b,0)/scores.length);nRows=totalN;}}
    const icVals=Object.values(platOps).map(od=>od.ic_avg).filter(Boolean);
    const ptVals=Object.values(platOps).map(od=>od.pt_avg).filter(Boolean);
    if(icVals.length) icAvg=icVals.reduce((a,b)=>a+b,0)/icVals.length;
    if(ptVals.length) ptAvg=ptVals.reduce((a,b)=>a+b,0)/ptVals.length;
  }}

  const sC=scColor(dispScore);
  const icLatStr=icAvg?fmtMs(icAvg):"—";
  const ptLatStr=ptAvg?fmtMs(ptAvg):"—";
  const opLabel=selOp?(OP_LABELS[selOp]||selOp):"综合均分";
  const pctBar=dispScore?Math.min(dispScore,100):0;
  let advHtml="";
  if(dispScore!=null){{
    const faster=dispScore>=100;
    const advC=faster?"#2e7d32":"#c62828";
    const advTxt=faster?`自研快 ${{dispScore-100}}%（延迟更低）`:`自研慢 ${{100-dispScore}}%（延迟更高）`;
    advHtml=`<div class="col-adv-wrap"><span class="col-adv" style="background:${{advC}}14;color:${{advC}};">${{advTxt}}</span></div>`;
  }}
  const opBarRows=(()=>{{
    return Object.keys(platOps).map(op=>{{
      const od=platOps[op]; const s=od?od.avg_score:null; const pct=s?Math.min(s,100):0;
      return `<div class="fw-row" style="margin-bottom:4px;">
        <span class="fw-row-name" style="color:${{c}};font-size:22px;width:160px;">${{OP_LABELS[op]||op}}</span>
        <div class="fw-bar-t"><div class="fw-bar-f" style="width:${{pct}}%;background:${{c}};"></div></div>
        <span class="fw-row-s" style="color:${{scColor(s)}};font-size:22px;width:60px;">${{s!=null?s:"—"}}</span>
      </div>`;
    }}).join("");
  }})();

  return `<div class="pcard" style="border-color:${{c}}55;" onclick="openModal('${{p.key}}','ops')">
    <div class="pcard-accent" style="background:${{c}};"></div>
    <div class="pcard-head"><div class="pcard-name">${{p.name}}</div>
      <div class="pcard-type" style="background:${{c}}18;color:${{c}};">${{p.type}}</div></div>
    ${{selOp?`
    <div class="pcols">
      <div class="pcol">
        <span class="fw-badge" style="background:${{c}}18;color:${{c}};">InfiniCore ✦</span>
        <div class="col-score" style="color:${{sC}};">${{dispScore!=null?dispScore:"—"}}</div>
        <div class="col-score-lbl">/ 100 最优得分</div>
        <div class="col-bar-t"><div class="col-bar-f" style="width:${{pctBar}}%;background:${{c}};"></div></div>
        <div><span class="col-peak" style="color:${{c}};">${{icLatStr}}</span></div>
        <div class="col-cfg">${{opLabel}}</div>
      </div>
      <div class="pcol-div"></div>
      <div class="pcol">
        <span class="fw-badge" style="background:#88888818;color:#888;">PyTorch</span>
        <div class="col-score" style="color:#555;">100</div>
        <div class="col-score-lbl">/ 100 基准</div>
        <div class="col-bar-t"><div class="col-bar-f" style="width:100%;background:#aaa;"></div></div>
        <div><span class="col-peak" style="color:#888;">${{ptLatStr}}</span></div>
        <div class="col-cfg">${{opLabel}}</div>
      </div>
    </div>
    ${{advHtml}}`
    :`<div style="padding:14px 18px 10px;">
        <div class="si-vs" style="font-size:24px;margin-bottom:10px;">InfiniCore vs PyTorch · 各算子均分</div>
        ${{opBarRows}}
      </div>`
    }}
    <div class="tap-h">点击查看全量数据 →</div>
  </div>`;
}}

/* ── INFER card ── */
function buildInferCard(p,c,dim){{
  // Apply batch / in-len filters
  const batchFilt = filt['batch']?parseFloat(filt['batch']):null;
  const inFilt    = filt['in']   ?parseFloat(filt['in'])   :null;
  const rows=INFER.filter(r=>r.platform===p.key
    &&(batchFilt==null||r.batch===batchFilt)
    &&(inFilt==null   ||r.in_tok===inFilt));
  if(!rows.length) return '';

  const nvRows=INFER.filter(r=>r.platform==="nvidia"
    &&(batchFilt==null||r.batch===batchFilt)
    &&(inFilt==null   ||r.in_tok===inFilt));
  const ttfts=rows.map(r=>r.il_ttft).filter(Boolean);
  const bestTTFT=ttfts.length?Math.min(...ttfts):null;

  // ── Same-config scoring: for each row find matching NVIDIA row ──────────
  let pfScores=[], dcScores=[], pfPairs=[], dcPairs=[];
  rows.forEach(r=>{{
    const nv=nvRows.find(n=>n.batch===r.batch&&n.in_tok===r.in_tok&&n.out_tok===r.out_tok);
    if(!nv) return;
    if(r.prefill_tps&&nv.prefill_tps){{
      const s=Math.round(r.prefill_tps/nv.prefill_tps*100);
      pfScores.push(s);
      pfPairs.push({{score:s,val:r.prefill_tps,cfg:`batch=${{r.batch}} in=${{r.in_tok}} out=${{r.out_tok}}`}});
    }}
    if(r.decode_tps&&nv.decode_tps){{
      const s=Math.round(r.decode_tps/nv.decode_tps*100);
      dcScores.push(s);
      dcPairs.push({{score:s,val:r.decode_tps,cfg:`batch=${{r.batch}} in=${{r.in_tok}} out=${{r.out_tok}}`}});
    }}
  }});

  // Best same-config score and its corresponding value
  const pfBest = pfPairs.length ? pfPairs.reduce((a,b)=>a.score>b.score?a:b) : null;
  const dcBest = dcPairs.length ? dcPairs.reduce((a,b)=>a.score>b.score?a:b) : null;
  const pfScore = pfBest ? pfBest.score : null;
  const dcScore = dcBest ? dcBest.score : null;
  const pfSC=scColor(pfScore), dcSC=scColor(dcScore);

  // Advantage badge: based on prefill best score
  let advHtml2="";
  if(pfScore!=null){{
    const diff=pfScore-100;
    const faster=diff>=0; const advC=faster?"#2e7d32":"#c62828";
    advHtml2=`<div class="col-adv-wrap"><span class="col-adv" style="background:${{advC}}14;color:${{advC}};">${{faster?"领先":"落后"}} ${{Math.abs(diff)}}% vs A100</span></div>`;
  }}

  return `<div class="pcard" style="border-color:${{c}}55;" onclick="openModal('${{p.key}}','infer')">
    <div class="pcard-accent" style="background:${{c}};"></div>
    <div class="pcard-head"><div class="pcard-name">${{p.name}}</div>
      <div class="pcard-type" style="background:${{c}}18;color:${{c}};">${{p.type}}</div></div>
    <div class="pcols">
      <div class="pcol">
        <span class="fw-badge" style="background:${{c}}18;color:${{c}};">Prefill ✦</span>
        <div class="col-score" style="color:${{pfSC}};">${{pfScore!=null?pfScore:"—"}}</div>
        <div class="col-score-lbl">/ 100 同配置 vs A100</div>
        <div class="col-bar-t"><div class="col-bar-f" style="width:${{Math.min(pfScore||0,100)}}%;background:${{c}};"></div></div>
        <div><span class="col-peak" style="color:${{c}};">${{pfBest?fmtV(pfBest.val):"—"}}</span><span class="col-unit">tok/s</span></div>
        <div class="col-cfg">${{pfBest?pfBest.cfg:""}} · ${{rows.length}}条</div>
      </div>
      <div class="pcol-div"></div>
      <div class="pcol">
        <span class="fw-badge" style="background:${{c}}28;color:${{c}};">Decode</span>
        <div class="col-score" style="color:${{dcSC}};">${{dcScore!=null?dcScore:"—"}}</div>
        <div class="col-score-lbl">/ 100 同配置 vs A100</div>
        <div class="col-bar-t"><div class="col-bar-f" style="width:${{Math.min(dcScore||0,100)}}%;background:${{c}};opacity:.7;"></div></div>
        <div><span class="col-peak" style="color:${{c}};">${{dcBest?fmtV(dcBest.val):"—"}}</span><span class="col-unit">tok/s</span></div>
        <div class="col-cfg">${{dcBest?dcBest.cfg:""}} · TTFT ${{bestTTFT?bestTTFT.toFixed(0)+"ms":"—"}}</div>
      </div>
    </div>
    ${{advHtml2}}
    <div class="tap-h">点击查看推理详情 →</div>
  </div>`;
}}

/* ── TRAIN card ── */
function buildTrainCard(p,c){{
  const fwFilt=filt['fw']||null;
  const rows=TRAIN.filter(r=>r.platform===p.key
    &&(fwFilt==null||r.framework.toLowerCase()===fwFilt.toLowerCase()));
  const nvRow=TRAIN.find(r=>r.platform==="nvidia"
    &&(fwFilt==null||r.framework.toLowerCase()===fwFilt.toLowerCase()));
  const nvTpps=nvRow?nvRow.tpps:null;

  if(!rows.length) return '';

  const best=Math.max(...rows.map(r=>r.tpps).filter(Boolean));
  const score=nvTpps?Math.round(best/nvTpps*100):null;
  const sC=scColor(score); const R=44,circ=2*Math.PI*R;
  const filled=score?(Math.min(score,100)/100*circ).toFixed(1):"0";
  const bestRow=rows.find(r=>r.tpps===best)||rows[0];

  return `<div class="pcard" style="border-color:${{c}}55;" onclick="openModal('${{p.key}}','train')">
    <div class="pcard-accent" style="background:${{c}};"></div>
    <div class="pcard-head"><div class="pcard-name">${{p.name}}</div>
      <div class="pcard-type" style="background:${{c}}18;color:${{c}};">${{p.type}}</div></div>
    <div class="psingle">
      <div class="ring-w"><svg viewBox="0 0 110 110">
        <circle fill="none" stroke="#f0f0f0" stroke-width="6" cx="55" cy="55" r="${{R}}"/>
        <circle fill="none" stroke="${{score!=null?c:"#eee"}}" stroke-width="6"
          stroke-linecap="round" cx="55" cy="55" r="${{R}}"
          stroke-dasharray="${{filled}} ${{circ.toFixed(1)}}"/></svg>
        <div class="ring-c"><span class="ring-s" style="color:${{sC}};">${{score!=null?score:"—"}}</span>
        <span class="ring-d">vs A100</span></div></div>
      <div class="si">
        <div class="si-vs">${{bestRow.framework}} · ${{bestRow.model}}</div>
        <div class="fw-row"><span class="fw-row-name" style="color:${{c}};font-size:11px;">吞吐</span>
          <div class="fw-bar-t"><div class="fw-bar-f" style="width:${{Math.min(score||0,100)}}%;background:${{c}};"></div></div>
          <span class="fw-row-s" style="color:#333;font-size:11px;">${{fmtV(best)}} t/s</span></div>
        <div class="si-cfg">${{bestRow.n_gpu}}GPU · ${{bestRow.seq_len}}seq · ${{bestRow.dtype}}</div>
      </div>
    </div>
    <div class="tap-h">点击查看训练详情 →</div>
  </div>`;
}}

/* ── COMM card ── */
function buildCommCard(p,c){{
  const ctFilt=filt['ct']||null;
  const rows=COMM.filter(r=>r.platform===p.key
    &&(ctFilt==null||r.comm_type===ctFilt));
  const nvRows=COMM.filter(r=>r.platform==="nvidia"
    &&(ctFilt==null||r.comm_type===ctFilt));

  if(!rows.length) return '';

  const fwRows=rows.map(r=>{{
    const nvMatch=nvRows.find(n=>n.comm_type===r.comm_type);
    const nvBw=nvMatch?nvMatch.bw_GBps:null;
    const score=nvBw?Math.round(r.bw_GBps/nvBw*100):null;
    const pct=score?Math.min(score,100):0;
    return `<div class="fw-row"><span class="fw-row-name" style="color:${{c}};font-size:11px;">${{r.comm_type}}</span>
      <div class="fw-bar-t"><div class="fw-bar-f" style="width:${{pct}}%;background:${{c}};"></div></div>
      <span class="fw-row-s" style="color:${{scColor(score)}};font-size:11px;">${{score!=null?score:"—"}}</span></div>`;
  }}).join("");

  const bestScore=Math.max(...rows.map(r=>{{
    const nv=nvRows.find(n=>n.comm_type===r.comm_type);
    return nv?Math.round(r.bw_GBps/nv.bw_GBps*100):0;
  }}));
  const sC=scColor(bestScore); const R=44,circ=2*Math.PI*R;
  const filled=(Math.min(bestScore,100)/100*circ).toFixed(1);

  return `<div class="pcard" style="border-color:${{c}}55;" onclick="openModal('${{p.key}}','comm')">
    <div class="pcard-accent" style="background:${{c}};"></div>
    <div class="pcard-head"><div class="pcard-name">${{p.name}}</div>
      <div class="pcard-type" style="background:${{c}}18;color:${{c}};">${{p.type}}</div></div>
    <div class="psingle">
      <div class="ring-w"><svg viewBox="0 0 110 110">
        <circle fill="none" stroke="#f0f0f0" stroke-width="6" cx="55" cy="55" r="${{R}}"/>
        <circle fill="none" stroke="${{c}}" stroke-width="6" stroke-linecap="round"
          cx="55" cy="55" r="${{R}}" stroke-dasharray="${{filled}} ${{circ.toFixed(1)}}"/></svg>
        <div class="ring-c"><span class="ring-s" style="color:${{sC}};">${{bestScore||"—"}}</span>
        <span class="ring-d">vs A100</span></div></div>
      <div class="si">
        <div class="si-vs">vs NVIDIA 带宽</div>
        ${{fwRows}}
      </div>
    </div>
    <div class="tap-h">点击查看通信详情 →</div>
  </div>`;
}}

/* ─────────────────────────────────────────────────────────────────────────────
   MODAL
───────────────────────────────────────────────────────────────────────────── */
function openModal(pk, mode){{
  const p=PLATS.find(x=>x.key===pk); const c=COLORS[pk]||"#888";
  mState={{pk,mode,activeTab:'data',activeDtype:null,activeOp:filt['op']||null,activeInferTab:'prefill'}};
  const bg=document.createElement("div"); bg.className="ov-bg"; bg.onclick=closeModal;
  const wrap=document.createElement("div"); wrap.className="ov-wrap";
  const mdiv=document.createElement("div"); mdiv.className="modal";
  wrap.appendChild(mdiv); document.body.appendChild(bg); document.body.appendChild(wrap);
  modalEl=wrap; modalBg=bg;
  window._mdiv=mdiv;
  renderModalContent(p,c);
}}
function closeModal(){{
  if(modalEl){{modalEl.remove();modalEl=null;}}
  if(modalBg){{modalBg.remove();modalBg=null;}}
}}
window.switchTab=function(t){{mState.activeTab=t; const p=PLATS.find(x=>x.key===mState.pk); renderModalContent(p,COLORS[mState.pk]||"#888");}}
window.setDtype=function(d){{mState.activeDtype=d; const p=PLATS.find(x=>x.key===mState.pk); renderModalContent(p,COLORS[mState.pk]||"#888");}}
window.setMOp=function(op){{mState.activeOp=op; const p=PLATS.find(x=>x.key===mState.pk); renderModalContent(p,COLORS[mState.pk]||"#888");}}
window.setInferTab=function(t){{mState.activeInferTab=t; const p=PLATS.find(x=>x.key===mState.pk); renderModalContent(p,COLORS[mState.pk]||"#888");}}

function renderModalContent(p,c){{
  const mdiv=window._mdiv; if(!mdiv) return;
  const {{mode,activeTab,activeDtype,activeOp,activeInferTab}}=mState;

  let headHtml="", bodyHtml="";

  if(mode==="ops"){{
    const platOps=OPS[p.key]||{{}};
    const opKeys=Object.keys(platOps);
    // Which op to show detail for
    const curOp=activeOp||opKeys[0];
    const od=platOps[curOp]||{{}};
    const allRows=od.rows||[];
    const filtered=activeDtype?allRows.filter(r=>r.dtype===activeDtype):allRows;
    const dtypes=[...new Set(allRows.map(r=>r.dtype).filter(Boolean))].sort();
    const avgSc=od.avg_score; const ic_avg=od.ic_avg; const pt_avg=od.pt_avg;
    const bestRow=od.best_row; const bestSc=od.best_row_score;
    const nAll=Object.values(platOps).reduce((a,o)=>a+o.n,0);

    const opBtns=['<button class="dtype-btn'+(activeOp===null?" on":"")+ '" onclick="setMOp(null)">全部算子</button>',
      ...opKeys.map(op=>`<button class="dtype-btn${{activeOp===op?" on":""}}" onclick="setMOp('${{op}}')">${{OP_LABELS[op]||op}} ${{platOps[op].avg_score!=null?platOps[op].avg_score:""}}</button>`)
    ].join("");

    headHtml=`
      <div class="modal-title-row">
        <div class="modal-dot" style="background:${{c}};"></div>
        <div class="modal-title" style="color:${{c}};">${{p.name}} · 算子详情</div>
        <button class="modal-close" onclick="closeModal()">✕</button>
      </div>
      <div class="modal-sub">InfiniCore vs PyTorch · ${{nAll}}条测试记录
        ${{avgSc!=null?`<b style="color:${{scColor(avgSc)}}"> 平均得分 ${{avgSc}}</b>`:""}}
        ${{bestSc&&bestSc>=100?` · <b style="color:#2e7d32">自研领先 ${{bestSc-100}}%</b>`:""}}
      </div>
      <div class="sum-row">
        <div class="sum-pill"><div class="sum-val" style="color:${{scColor(avgSc)}}">${{avgSc!=null?avgSc:"—"}}</div><div class="sum-lbl">平均得分</div></div>
        <div class="sum-pill"><div class="sum-val" style="color:#333">${{nAll}}</div><div class="sum-lbl">测试条数</div></div>
        <div class="sum-pill"><div class="sum-val" style="color:${{c}};font-size:18px">${{ic_avg?fmtMs(ic_avg):"—"}}</div><div class="sum-lbl">InfiniCore 均值</div></div>
        <div class="sum-pill"><div class="sum-val" style="color:#888;font-size:18px">${{pt_avg?fmtMs(pt_avg):"—"}}</div><div class="sum-lbl">PyTorch 均值</div></div>
      </div>`;

    let scoreBody="", dataBody="";
    // score tab
    const scoreRows=opKeys.map(op=>{{
      const od2=platOps[op]; const s=od2.avg_score;
      const sC2=scColor(s); const pct=s?Math.min(s,100):0;
      const concl=s==null?"—":s>=100?`<span class="pct-badge" style="background:#e8f5e9;color:#2e7d32">自研更快 +${{s-100}}%</span>`:`<span class="pct-badge" style="background:#fce4ec;color:#c62828">开源更快</span>`;
      return `<tr>
        <td><b>${{OP_LABELS[op]||op}}</b></td>
        <td style="color:${{c}};font-weight:800">${{od2.ic_avg?fmtMs(od2.ic_avg):"—"}}</td>
        <td style="color:#888">${{od2.pt_avg?fmtMs(od2.pt_avg):"—"}}</td>
        <td><b style="font-size:15px;color:${{sC2}}">${{s!=null?s:"—"}}</b> / 100</td>
        <td>${{od2.n}}</td><td>${{concl}}</td></tr>`;
    }}).join("");
    const bestRowHtml=bestRow?`<div class="best-row-card">
      <b>🏆 最佳配置对比（同 shape × dtype）</b><br>
      Shape: <code>${{bestRow.shape}}</code> · 精度: <b>${{bestRow.dtype}}</b><br>
      InfiniCore: <b style="color:${{c}}">${{fmtMs(bestRow.ic_lat)}}</b> &nbsp;
      PyTorch: ${{fmtMs(bestRow.pt_lat)}} &nbsp;
      得分: <b style="color:#2e7d32">${{bestSc}}</b>（自研快 ${{bestSc-100}}%）
    </div>`:"";
    scoreBody=`
      ${{bestRowHtml}}
      <div class="score-explain">得分说明：每行得分 = 同行 PyTorch延迟 ÷ InfiniCore延迟 × 100（同 shape + dtype）· >100 自研更快 · 平均得分 = 所有行均值 · 完整计分体系待定</div>
      <table class="score-tbl"><thead><tr><th>算子</th><th>InfiniCore 均值</th><th>PyTorch 均值</th><th>平均得分</th><th>条数</th><th>结论</th></tr></thead>
      <tbody>${{scoreRows}}</tbody></table>`;

    // data tab
    const maxLat=Math.max(...filtered.map(r=>Math.max(r.ic_lat||0,r.pt_lat||0)).filter(x=>x>0),1);
    const dataRows=filtered.map(r=>{{
      const s=r.ic_lat&&r.pt_lat?Math.round(r.pt_lat/r.ic_lat*100):null;
      const sC2=scColor(s); const arrow=s!=null?(s>=100?"↑":"↓"):"";
      const icPct=r.ic_lat?Math.round(r.ic_lat/maxLat*100):0;
      const ptPct=r.pt_lat?Math.round(r.pt_lat/maxLat*100):0;
      const dcol=DTYPE_COLORS[r.dtype]||"#888";
      return `<tr>
        <td class="shape-cell">${{curOp!==activeOp?`${{OP_LABELS[op]||op}} | `:""}}${{r.shape||"—"}}</td>
        <td><span class="dtype-pill" style="background:${{dcol}}18;color:${{dcol}}">${{r.dtype}}</span></td>
        <td style="color:${{c}};font-weight:800">${{r.ic_lat!=null?fmtMs(r.ic_lat):"待测"}}</td>
        <td style="color:#888">${{r.pt_lat!=null?fmtMs(r.pt_lat):"待测"}}</td>
        <td><span class="score-pill" style="background:${{sC2}}18;color:${{sC2}}">${{s!=null?s+arrow:"—"}}</span></td>
        <td><div class="dual-bar">
          <div class="dual-row"><span class="dual-lbl" style="color:${{c}}">自研</span>
            <div class="dual-track"><div class="dual-fill" style="width:${{icPct}}%;background:${{c}}"></div></div>
            <span class="dual-ms" style="color:${{c}}">${{r.ic_lat!=null?fmtMs(r.ic_lat):""}}</span></div>
          <div class="dual-row"><span class="dual-lbl" style="color:#aaa">开源</span>
            <div class="dual-track"><div class="dual-fill" style="width:${{ptPct}}%;background:#aaa"></div></div>
            <span class="dual-ms" style="color:#aaa">${{r.pt_lat!=null?fmtMs(r.pt_lat):""}}</span></div>
        </div></td></tr>`;
    }}).join("");
    dataBody=`
      <div class="dtype-row"><span class="dtype-lbl">精度：</span>
        <button class="dtype-btn${{!activeDtype?" on":""}}" onclick="setDtype(null)">全部</button>
        ${{dtypes.map(d=>`<button class="dtype-btn${{activeDtype===d?" on":""}}" onclick="setDtype('${{d}}')">${{d}}</button>`).join("")}}
      </div>
      <div class="notice">延迟越低越好 · 得分 = 同行 PyTorch延迟 ÷ InfiniCore延迟 × 100 · ✦ InfiniCore 为自研</div>
      <table class="dtbl"><thead><tr><th>Shape 配置</th><th>精度</th><th>InfiniCore ✦</th><th>PyTorch</th><th>得分</th><th>延迟对比</th></tr></thead>
      <tbody>${{dataRows||'<tr><td colspan="6" style="text-align:center;padding:20px;color:#ccc">暂无数据</td></tr>'}}</tbody></table>`;

    const tabs=`<div class="modal-tabs">
      <div class="mtab${{activeTab==="data"?" on":""}}" onclick="switchTab('data')">📋 数据明细</div>
      <div class="mtab${{activeTab==="score"?" on":""}}" onclick="switchTab('score')">📊 得分说明</div>
    </div>`;
    const opBar=`<div class="dtype-row" style="padding:12px 26px 0;">${{opBtns}}</div>`;
    mdiv.innerHTML=`<div class="modal-head">${{headHtml}}${{opBar}}</div>${{tabs}}
      <div class="modal-body">${{activeTab==="data"?dataBody:scoreBody}}</div>`;

  }}else if(mode==="infer"){{
    const bF=mState.batchFilt||null, iF=mState.inFilt||null;
  const rows=INFER.filter(r=>r.platform===p.key
    &&(bF==null||r.batch===bF)&&(iF==null||r.in_tok===iF));
  const nvRows=INFER.filter(r=>r.platform==="nvidia"
    &&(bF==null||r.batch===bF)&&(iF==null||r.in_tok===iF));
    const pfTps=rows.map(r=>r.prefill_tps).filter(Boolean);
    const dcTps=rows.map(r=>r.decode_tps).filter(Boolean);
    const ttfts=rows.map(r=>r.il_ttft).filter(Boolean);

    headHtml=`<div class="modal-title-row">
      <div class="modal-dot" style="background:${{c}};"></div>
      <div class="modal-title" style="color:${{c}};">${{p.name}} · 推理详情</div>
      <button class="modal-close" onclick="closeModal()">✕</button>
    </div>
    <div class="modal-sub">InfiniLM · ${{rows.length}}条测试记录 · 9G8B 模型</div>
    <div class="sum-row">
      <div class="sum-pill"><div class="sum-val" style="color:${{c}};font-size:20px">${{pfTps.length?fmtV(Math.max(...pfTps)):"—"}}</div><div class="sum-lbl">Prefill 峰值 TPS</div></div>
      <div class="sum-pill"><div class="sum-val" style="color:${{c}};font-size:20px">${{dcTps.length?fmtV(Math.max(...dcTps)):"—"}}</div><div class="sum-lbl">Decode 峰值 TPS</div></div>
      <div class="sum-pill"><div class="sum-val" style="color:#888;font-size:20px">${{ttfts.length?Math.min(...ttfts).toFixed(1)+"ms":"—"}}</div><div class="sum-lbl">最低 TTFT</div></div>
      <div class="sum-pill"><div class="sum-val" style="color:#333">${{rows.length}}</div><div class="sum-lbl">测试条数</div></div>
    </div>`;

    const itabHtml=`<div class="modal-tabs">
      <div class="mtab${{activeInferTab==="prefill"?" on":""}}" onclick="setInferTab('prefill')">📈 Prefill</div>
      <div class="mtab${{activeInferTab==="decode"?" on":""}}" onclick="setInferTab('decode')">🔄 Decode</div>
      <div class="mtab${{activeInferTab==="all"?" on":""}}" onclick="setInferTab('all')">📋 全量明细</div>
    </div>`;

    let dataHtml="";
    if(activeInferTab==="prefill"){{
      // Group by batch, show prefill_tps vs input_tokens
      const batchGroups={{}};
      rows.forEach(r=>{{(batchGroups[r.batch]=batchGroups[r.batch]||[]).push(r);}});
      const tblRows=rows.sort((a,b)=>a.batch-b.batch||a.in_tok-b.in_tok).map(r=>{{
        const nvMatch=nvRows.find(n=>n.batch===r.batch&&n.in_tok===r.in_tok&&n.out_tok===r.out_tok);
        const nvPf=nvMatch?nvMatch.prefill_tps:null;
        const pct=nvPf&&r.prefill_tps?Math.round(r.prefill_tps/nvPf*100):null;
        return `<tr>
          <td>batch=${{r.batch}} in=${{r.in_tok}} out=${{r.out_tok}}</td>
          <td class="${{r.prefill_tps?"val-good":"val-na"}}">${{r.prefill_tps?fmtV(r.prefill_tps):"—"}}</td>
          <td class="val-na">${{nvPf?fmtV(nvPf):"—"}}</td>
          <td>${{pct!=null?`<span class="pct-badge" style="background:${{scColor(pct)}}18;color:${{scColor(pct)}}">${{pct}}%</span>`:"—"}}</td>
          <td style="color:#888;font-size:11px">${{r.il_ttft?r.il_ttft.toFixed(1)+"ms":"—"}}</td>
        </tr>`;
      }}).join("");
      dataHtml=`<div class="notice">Prefill 吞吐量（tokens/s）· 数据越高越好 · vs NVIDIA InfiniLM 同配置</div>
        <table class="itbl"><thead><tr><th>配置</th><th>Prefill TPS</th><th>A100 基线</th><th>vs A100</th><th>TTFT</th></tr></thead>
        <tbody>${{tblRows}}</tbody></table>`;
    }}else if(activeInferTab==="decode"){{
      const tblRows=rows.sort((a,b)=>a.batch-b.batch||a.in_tok-b.in_tok).map(r=>{{
        const nvMatch=nvRows.find(n=>n.batch===r.batch&&n.in_tok===r.in_tok&&n.out_tok===r.out_tok);
        const nvDc=nvMatch?nvMatch.decode_tps:null;
        const pct=nvDc&&r.decode_tps?Math.round(r.decode_tps/nvDc*100):null;
        return `<tr>
          <td>batch=${{r.batch}} in=${{r.in_tok}} out=${{r.out_tok}}</td>
          <td class="${{r.decode_tps?"val-good":"val-na"}}">${{r.decode_tps?fmtV(r.decode_tps):"—"}}</td>
          <td class="val-na">${{nvDc?fmtV(nvDc):"—"}}</td>
          <td>${{pct!=null?`<span class="pct-badge" style="background:${{scColor(pct)}}18;color:${{scColor(pct)}}">${{pct}}%</span>`:"—"}}</td>
          <td style="color:#888;font-size:11px">${{r.il_dec?r.il_dec.toFixed(2)+"ms":"—"}}</td>
        </tr>`;
      }}).join("");
      dataHtml=`<div class="notice">Decode 吞吐量（tokens/s）· 越高越好 · vs NVIDIA 同配置</div>
        <table class="itbl"><thead><tr><th>配置</th><th>Decode TPS</th><th>A100 基线</th><th>vs A100</th><th>Decode延迟</th></tr></thead>
        <tbody>${{tblRows}}</tbody></table>`;
    }}else{{
      const tblRows=rows.sort((a,b)=>a.batch-b.batch||a.in_tok-b.in_tok).map(r=>
        `<tr><td>${{r.batch}}</td><td>${{r.in_tok}}</td><td>${{r.out_tok}}</td><td>${{r.dtype}}</td>
         <td class="${{r.il_ttft?"val-good":"val-na"}}">${{r.il_ttft!=null?r.il_ttft.toFixed(1)+"ms":"—"}}</td>
         <td class="${{r.prefill_tps?"val-good":"val-na"}}">${{r.prefill_tps?fmtV(r.prefill_tps):"—"}}</td>
         <td class="${{r.decode_tps?"val-good":"val-na"}}">${{r.decode_tps?fmtV(r.decode_tps):"—"}}</td>
         <td style="color:#888;font-size:11px">${{r.il_dec?r.il_dec.toFixed(2)+"ms":"—"}}</td></tr>`
      ).join("");
      dataHtml=`<table class="itbl"><thead><tr><th>Batch</th><th>In-len</th><th>Out-len</th><th>dtype</th>
        <th>TTFT</th><th>Prefill TPS</th><th>Decode TPS</th><th>Decode延迟</th></tr></thead>
        <tbody>${{tblRows}}</tbody></table>`;
    }}

    mdiv.innerHTML=`<div class="modal-head">${{headHtml}}</div>${{itabHtml}}
      <div class="modal-body">${{dataHtml}}</div>`;

  }}else if(mode==="train"){{
    const fwF=filt['fw']||null;
  const rows=TRAIN.filter(r=>r.platform===p.key
    &&(fwF==null||r.framework.toLowerCase()===fwF.toLowerCase()));
  const nvRow=TRAIN.find(r=>r.platform==="nvidia"
    &&(fwF==null||r.framework.toLowerCase()===fwF.toLowerCase()));
    headHtml=`<div class="modal-title-row">
      <div class="modal-dot" style="background:${{c}};"></div>
      <div class="modal-title" style="color:${{c}};">${{p.name}} · 训练详情</div>
      <button class="modal-close" onclick="closeModal()">✕</button>
    </div><div class="modal-sub">训练吞吐 · tokens per process per second</div>`;
    const tblRows=rows.map(r=>{{
      const score=nvRow?Math.round(r.tpps/nvRow.tpps*100):null;
      return `<tr><td>${{r.framework}}</td><td>${{r.model}}</td>
        <td>${{r.n_gpu}} GPU · seq ${{r.seq_len}}</td><td>${{r.dtype}}</td>
        <td>flash: ${{r.flash_attn}}</td>
        <td><b style="color:${{c}}">${{fmtV(r.tpps)}}</b> t/s</td>
        <td>${{nvRow?`<b style="color:#888">${{fmtV(nvRow.tpps)}} t/s</b>`:"—"}}</td>
        <td>${{score!=null?`<span class="pct-badge" style="background:${{scColor(score)}}18;color:${{scColor(score)}}">${{score}}%</span>`:"—"}}</td>
        <td style="color:#888;font-size:11px">${{r.remarks}}</td></tr>`;
    }}).join("");
    mdiv.innerHTML=`<div class="modal-head">${{headHtml}}</div>
      <div class="modal-body"><div class="notice">训练吞吐 tpps = tokens per process per second · vs NVIDIA Megatron 基线</div>
        <table class="gen-tbl"><thead><tr><th>框架</th><th>模型</th><th>并行配置</th><th>精度</th><th>Flash Attn</th><th>吞吐</th><th>A100 基线</th><th>vs A100</th><th>备注</th></tr></thead>
        <tbody>${{tblRows||'<tr><td colspan="9" style="text-align:center;padding:20px;color:#ccc">暂无数据</td></tr>'}}</tbody></table></div>`;

  }}else if(mode==="comm"){{
    const ctF=filt['ct']||null;
  const rows=COMM.filter(r=>r.platform===p.key
    &&(ctF==null||r.comm_type===ctF));
  const nvRows=COMM.filter(r=>r.platform==="nvidia"
    &&(ctF==null||r.comm_type===ctF));
    headHtml=`<div class="modal-title-row">
      <div class="modal-dot" style="background:${{c}};"></div>
      <div class="modal-title" style="color:${{c}};">${{p.name}} · 通信详情</div>
      <button class="modal-close" onclick="closeModal()">✕</button>
    </div><div class="modal-sub">集合通信带宽 · GB/s</div>`;
    const tblRows=rows.map(r=>{{
      const nvMatch=nvRows.find(n=>n.comm_type===r.comm_type);
      const score=nvMatch?Math.round(r.bw_GBps/nvMatch.bw_GBps*100):null;
      return `<tr><td>${{r.link_type}}</td><td>${{r.comm_type}}</td>
        <td>${{r.n_gpu!=null?r.n_gpu+"GPU":"—"}}</td>
        <td><b style="color:${{c}}">${{r.bw_GBps}} GB/s</b></td>
        <td>${{nvMatch?`${{nvMatch.bw_GBps}} GB/s`:"—"}}</td>
        <td>${{score!=null?`<span class="pct-badge" style="background:${{scColor(score)}}18;color:${{scColor(score)}}">${{score}}%</span>`:"—"}}</td>
        <td style="color:#888;font-size:11px">${{r.remarks}}</td></tr>`;
    }}).join("");
    mdiv.innerHTML=`<div class="modal-head">${{headHtml}}</div>
      <div class="modal-body"><div class="notice">带宽 vs NVIDIA NVLink 基线 · 单向带宽</div>
        <table class="gen-tbl"><thead><tr><th>Link类型</th><th>通信类型</th><th>GPU数</th><th>带宽</th><th>A100基线</th><th>vs A100</th><th>备注</th></tr></thead>
        <tbody>${{tblRows||'<tr><td colspan="7" style="text-align:center;padding:20px;color:#ccc">暂无数据</td></tr>'}}</tbody></table></div>`;
  }}

  // Prevent click propagation inside modal
  mdiv.onclick=e=>e.stopPropagation();
}}

function setDim(i){{aDim=i;filt={{}};renderTabs();renderFilters();renderCards();}}

renderTabs(); renderFilters(); renderCards();
</script>
</body></html>"""


# ── Main app ──────────────────────────────────────────────────────────────────
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
        run_id_filter = st.text_input("🔍 Run ID 模糊搜索", placeholder="输入关键词筛选")
    render_dashboard(run_id_filter)


def render_dashboard(run_id_filter=""):
    logo_path = Path(__file__).parent / "static" / "logos" / "logo.png"
    logo_html = ""
    if logo_path.exists():
        try:
            b64 = base64.b64encode(logo_path.read_bytes()).decode()
            logo_html = (
                f'<img src="data:image/png;base64,{b64}" '
                f'style="height:60px;vertical-align:middle;margin-right:14px;"/>'
            )
        except:
            pass

    st.markdown(
        f"""
    <div style="text-align:center;padding:32px 0 20px;">
        <div style="font-size:2.4em;font-weight:800;letter-spacing:.01em;
                    display:flex;align-items:center;justify-content:center;color:#1a1a2e;">
            {logo_html}InfiniBench 国产算力性能基准平台
        </div>
        <div style="font-size:1.0em;color:#666;margin-top:8px;letter-spacing:.05em;">
            统一评测 · 客观对标 · 加速国产 AI 基础设施成熟
        </div>
    </div>""",
        unsafe_allow_html=True,
    )

    try:
        all_runs = st.session_state.data_loader.list_test_runs()
        ci_summaries = st.session_state.data_loader.load_summaries()
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        st.code(traceback.format_exc())
        return

    if run_id_filter:
        all_runs = [r for r in all_runs if run_id_filter in r.get("run_id", "")]

    def _acc_of(r):
        al = list(r.get("accelerator_types", []) or [])
        if al:
            return al[0]
        return _infer_acc(r.get("run_id", "") or r.get("path", ""))

    active_accs = {_acc_of(r) for r in all_runs if _acc_of(r)}

    # ── Platform selector ──────────────────────────────────────────────────────
    cols = st.columns(len(PLATFORMS))
    for idx, p in enumerate(PLATFORMS):
        with cols[idx]:
            is_sel = p["key"] in st.session_state.selected_platform_keys
            has_data = p["key"] in active_accs
            border_color = (
                PLATFORM_COLORS.get(p["key"], "#aaa") if is_sel else "#e0e0e0"
            )
            bg_color = "#f0f7ff" if is_sel else "#fafafa"
            shadow = f"0 3px 14px {border_color}55" if is_sel else "none"
            opacity = "1" if has_data else "0.45"
            status_html = (
                f'<div style="margin-top:8px;font-size:1.05em;font-weight:700;'
                f'color:{PLATFORM_COLORS.get(p["key"],"#333")};">✔ 已选</div>'
                if is_sel
                else '<div style="margin-top:8px;font-size:.9em;color:#ccc;">— 未选 —</div>'
            )
            st.markdown(
                f"""
            <div style="border:2.5px solid {border_color};border-radius:14px;
                        padding:18px 6px 14px;text-align:center;background:{bg_color};
                        min-height:195px;box-shadow:{shadow};cursor:pointer;opacity:{opacity};">
                {_logo_img_tag(p['logo'],size_px=108)}
                <div style="font-size:1.05em;font-weight:700;color:#222;margin-top:4px;">{p['label']}</div>
                {status_html}
            </div>""",
                unsafe_allow_html=True,
            )
            if st.button(" ", key=f"plat_{p['key']}", use_container_width=True):
                keys = list(st.session_state.selected_platform_keys)
                if p["key"] in keys:
                    keys.remove(p["key"])
                else:
                    keys.append(p["key"])
                st.session_state.selected_platform_keys = keys
                st.rerun()

    selected_accs = st.session_state.selected_platform_keys or list(active_accs)
    st.markdown("<div style='margin-top:6px;'></div>", unsafe_allow_html=True)
    st.divider()

    st.markdown(
        """
    <div style="font-size:1.9em;font-weight:900;color:#1a1a2e;margin-bottom:4px;">
        📊 多维度跨平台性能对比
    </div>
    <div style="font-size:1.0em;color:#888;margin-bottom:18px;font-weight:500;">
        选择上方平台卡片进行筛选 · 切换维度标签 · ✦ 为自研框架 · 点击卡片查看详情
    </div>""",
        unsafe_allow_html=True,
    )

    html = _build_dashboard_html(
        selected_accs=selected_accs,
        colors=PLATFORM_COLORS,
        label_map=PLATFORM_LABEL_MAP,
        sub_map=PLATFORM_SUB_MAP,
    )
    n = len(selected_accs)
    height = max(840, 220 + (n // 2 + 1) * 360)
    components.html(html, height=height, scrolling=False)
    st.divider()

    with st.expander("📈 CI 运行统计", expanded=False):
        render_ci_stats(ci_summaries)
    with st.expander("🧾 Dispatcher 汇总", expanded=False):
        render_dispatcher_summary(ci_summaries)
    with st.expander("📋 CI 详细记录", expanded=False):
        render_ci_detailed_table(ci_summaries)
    with st.expander("🔍 失败详情", expanded=False):
        render_failure_details(ci_summaries)

    st.markdown(
        '<div style="text-align:center;padding:20px 0 8px;color:#ccc;font-size:.85em;">'
        "InfiniBench · 国产算力统一性能基准平台 · 数据持续更新中</div>",
        unsafe_allow_html=True,
    )


# ── CI helpers (unchanged) ────────────────────────────────────────────────────
def render_ci_stats(ci_summaries):
    if not ci_summaries:
        st.info("暂无 CI 汇总数据")
        return
    tc = sum(s.get("total_tests", 0) for s in ci_summaries)
    ts = sum(s.get("successful_tests", 0) for s in ci_summaries)
    tf = sum(s.get("failed_tests", 0) for s in ci_summaries)
    avg = ts / tc * 100 if tc > 0 else 0
    r10 = ci_summaries[:10]
    r10s = sum(s.get("successful_tests", 0) for s in r10)
    r10t = sum(s.get("total_tests", 0) for s in r10)
    r10r = r10s / r10t * 100 if r10t > 0 else 0
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("CI运行次数", len(ci_summaries))
    c2.metric("测试用例总数", f"{tc:,}")
    c3.metric("通过用例", f"{ts:,}")
    c4.metric("失败用例", f"{tf:,}")
    c5.metric("平均成功率", f"{avg:.1f}%")
    c6.metric("最近10次", f"{r10r:.1f}%", delta=f"{r10r-avg:.1f}%")
    daily = {}
    for s in ci_summaries:
        dt = parse_timestamp(s.get("timestamp", ""))
        if not dt:
            continue
        dk = dt.strftime("%Y-%m-%d")
        daily.setdefault(dk, {"t": 0, "s": 0})
        daily[dk]["t"] += s.get("total_tests", 0)
        daily[dk]["s"] += s.get("successful_tests", 0)
    if daily:
        dates = sorted(daily.keys())
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=dates,
                y=[daily[d]["t"] for d in dates],
                name="总数",
                marker_color="lightblue",
            )
        )
        fig.add_trace(
            go.Bar(
                x=dates,
                y=[daily[d]["s"] for d in dates],
                name="成功",
                marker_color="lightgreen",
            )
        )
        fig.add_trace(
            go.Bar(
                x=dates,
                y=[daily[d]["t"] - daily[d]["s"] for d in dates],
                name="失败",
                marker_color="lightcoral",
            )
        )
        fig.update_layout(
            barmode="group", template="plotly_white", height=300, xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)


def render_dispatcher_summary(ci_summaries):
    if not ci_summaries:
        return
    rows = []
    for s in ci_summaries[:15]:
        t = s.get("total_tests", 0)
        rows.append(
            {
                "时间": format_time(s.get("timestamp", "")),
                "总测试数": t,
                "成功": s.get("successful_tests", 0),
                "失败": s.get("failed_tests", 0),
                "成功率": f"{s.get('successful_tests',0)/t*100:.1f}%" if t > 0 else "-",
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
        t = s.get("total_tests", 0)
        sc2 = s.get("successful_tests", 0)
        f = s.get("failed_tests", 0)
        dur = s.get("duration", s.get("total_duration_seconds", 0))
        res = s.get("results", [])
        git = s.get("git", {})
        sc_ = git.get("short_commit", "")
        cm = git.get("commit_message", "")
        br = git.get("branch", "")
        rows.append(
            {
                "时间": format_time(s.get("timestamp", "")),
                "Run ID": res[0].get("run_id", "-") if res else "-",
                "Commit": sc_[:8] if sc_ and sc_ != "unknown" else "本地",
                "提交信息": (cm[:50] + "...") if len(cm) > 50 else cm or "-",
                "分支": br if br and not br.startswith("unknown") else "local",
                "作者": git.get("commit_author", "-"),
                "总数": t,
                "✅": sc2,
                "❌": f,
                "成功率": f"{sc2/t*100:.1f}%" if t > 0 else "-",
                "状态": "✅ 成功" if f == 0 else "❌ 失败",
                "时长": f"{dur:.1f}s" if dur > 0 else "-",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_failure_details(ci_summaries):
    recs = []
    for s in ci_summaries:
        if not s.get("failed_tests"):
            continue
        git = s.get("git", {})
        sc_ = git.get("short_commit", "")
        details = s.get("failed_tests_details", []) or [
            {
                "testcase": r.get("testcase", "?"),
                "run_id": r.get("run_id", "?"),
                "result_code": r.get("result_code", -1),
            }
            for r in s.get("results", [])
            if r.get("result_code", 0) != 0
        ]
        if details:
            recs.append(
                {
                    "时间": format_time(s.get("timestamp", "")),
                    "失败数": len(details),
                    "详情": details,
                    "Commit": sc_[:8] if sc_ and sc_ != "unknown" else "本地",
                }
            )
    if not recs:
        st.info("暂无失败记录")
        return
    for rec in recs[:10]:
        with st.expander(f"📅 {rec['时间']} — {rec['失败数']} 个失败 (Commit: {rec['Commit']})"):
            for i, f in enumerate(rec["详情"][:15]):
                st.markdown(
                    f"**{i+1}.** `{f.get('testcase','?')}` — Run: `{f.get('run_id','?')}` — Code: {f.get('result_code',-1)}"
                )


if __name__ == "__main__":
    main()
