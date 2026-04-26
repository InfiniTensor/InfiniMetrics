#!/usr/bin/env bash
# =============================================================================
# InfiniBench Batch Runner — 全量批跑所有维度配置
#
# Usage:
#   ./run_batch.sh                    # 跑全部 5 个维度
#   ./run_batch.sh infer              # 只跑推理（infinilm + vllm）
#   ./run_batch.sh infer comm         # 跑推理 + 通信
#   ./run_batch.sh --skip-missing     # 模型路径不存在时自动跳过（推荐加上）
#
# 组合示例：
#   ./run_batch.sh infer --skip-missing
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS_DIR="${SCRIPT_DIR}/configs"
LOG_DIR="${SCRIPT_DIR}/logs/batch_$(date +%Y%m%d_%H%M%S)"
SUMMARY_LOG="${LOG_DIR}/summary.log"
PYTHON="${PYTHON:-python}"
mkdir -p "${LOG_DIR}"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*" | tee -a "${SUMMARY_LOG}"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*" | tee -a "${SUMMARY_LOG}"; }
error()   { echo -e "${RED}[FAIL]${NC}  $*" | tee -a "${SUMMARY_LOG}"; }
section() { printf "\n${BOLD}%s\n%-40s\n%s${NC}\n" \
            "════════════════════════════════════════" \
            "  $*" \
            "════════════════════════════════════════" | tee -a "${SUMMARY_LOG}"; }

# ── 参数解析 ──────────────────────────────────────────────────────────────────
SKIP_MISSING=false
CATEGORIES=()
for arg in "$@"; do
    case "$arg" in
        --skip-missing) SKIP_MISSING=true ;;
        infer|comm|hardware|training|operator) CATEGORIES+=("$arg") ;;
        *) warn "Unknown argument: $arg" ;;
    esac
done
[ ${#CATEGORIES[@]} -eq 0 ] && CATEGORIES=("infer" "comm" "hardware" "training" "operator")

# ── model_path 存在性检查 ─────────────────────────────────────────────────────
model_path_exists() {
    local mp
    mp=$(python3 -c "
import json, sys
try:
    d = json.load(open('$1'))
    if isinstance(d, list): d = d[0]
    print(d.get('config', {}).get('model_path', ''))
except: print('')
" 2>/dev/null)
    [ -z "$mp" ] && return 0   # 没有 model_path 字段（comm/hardware），直接放行
    [ -d "$mp" ]
}

# ── 单个 config 执行 ──────────────────────────────────────────────────────────
run_config() {
    local config_file="$1"
    local category="$2"
    local bname
    bname=$(basename "${config_file}" .json)
    local log_file="${LOG_DIR}/${category}_${bname}.log"

    # 读出 run_id 供打印
    local run_id
    run_id=$(python3 -c "
import json; d=json.load(open('${config_file}'))
if isinstance(d,list): d=d[0]
print(d.get('run_id', '${bname}'))
" 2>/dev/null)

    if $SKIP_MISSING && ! model_path_exists "${config_file}"; then
        warn "SKIP  ${run_id}  (model not found)"
        echo "SKIP|${run_id}" >> "${SUMMARY_LOG}"
        return 0
    fi

    echo -n "  Running ${run_id} ... "
    local start elapsed exit_code=0
    start=$(date +%s)

    ${PYTHON} main.py "${config_file}" >> "${log_file}" 2>&1 || exit_code=$?
    elapsed=$(( $(date +%s) - start ))

    if [ "${exit_code}" -eq 0 ]; then
        info "✅  (${elapsed}s)"
        echo "PASS|${run_id}|${elapsed}s" >> "${SUMMARY_LOG}"
    else
        error "❌  (${elapsed}s, exit=${exit_code}) → ${log_file}"
        echo "FAIL|${run_id}|${elapsed}s|exit=${exit_code}" >> "${SUMMARY_LOG}"
    fi
}

# ── 分类执行 ──────────────────────────────────────────────────────────────────
run_category() {
    local category="$1"
    local config_path="${CONFIGS_DIR}/${category}"
    [ ! -d "${config_path}" ] && { warn "No config dir: ${config_path}"; return; }

    local configs=()
    while IFS= read -r -d '' f; do configs+=("$f"); done \
        < <(find "${config_path}" -name "*.json" -print0 | sort -z)

    [ ${#configs[@]} -eq 0 ] && { warn "No configs in ${config_path}"; return; }

    section "${category^^}  (${#configs[@]} configs)"

    for config_file in "${configs[@]}"; do
        run_config "${config_file}" "${category}"
    done
}

# ── Main ──────────────────────────────────────────────────────────────────────
section "InfiniBench Batch Runner"
info "Time       : $(date)"
info "Categories : ${CATEGORIES[*]}"
info "Skip miss  : ${SKIP_MISSING}"
info "Log dir    : ${LOG_DIR}"

total_start=$(date +%s)
for cat in "${CATEGORIES[@]}"; do run_category "${cat}"; done
total_elapsed=$(( $(date +%s) - total_start ))

# 统计
total_pass=$(grep -c "^PASS|" "${SUMMARY_LOG}" 2>/dev/null || true)
total_fail=$(grep -c "^FAIL|" "${SUMMARY_LOG}" 2>/dev/null || true)
total_skip=$(grep -c "^SKIP|" "${SUMMARY_LOG}" 2>/dev/null || true)

section "Summary"
info "Total time : ${total_elapsed}s"
info "✅ PASS    : ${total_pass}"
info "❌ FAIL    : ${total_fail}"
info "⏭  SKIP    : ${total_skip}"
info "Log dir    : ${LOG_DIR}"

if [ "${total_fail:-0}" -gt 0 ]; then
    warn "Failed tests:"
    grep "^FAIL|" "${SUMMARY_LOG}" | while IFS='|' read -r _ name dur reason; do
        warn "  ❌ ${name}  ${dur}  ${reason}"
    done
    exit 1
fi
