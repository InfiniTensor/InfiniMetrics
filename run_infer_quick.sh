#!/usr/bin/env bash
# =============================================================================
# InfiniBench Quick Infer Validator
# 每个模型只跑 batch=1，快速验证能否跑通，再决定是否全量跑
#
# Usage:
#   ./run_infer_quick.sh              # infinilm + vllm 全部模型
#   ./run_infer_quick.sh infinilm     # 只验证 InfiniLM
#   ./run_infer_quick.sh vllm         # 只验证 vLLM
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
LOG_DIR="${SCRIPT_DIR}/logs/quick_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'
info()    { echo -e "${GREEN}[PASS]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[SKIP]${NC}  $*"; }
error()   { echo -e "${RED}[FAIL]${NC}  $*"; }
section() { echo -e "\n${BOLD}── $* ──${NC}"; }

# 实际已建软链接的模型 tag（对应 configs/infer/{fw}/ 文件名中的 tag 部分）
MODELS=(
    "qwen3_1_7b"
    "qwen3_4b"
    "qwen3_8b"
    "qwen2_5_7b_instruct"
    "qwen2_5_14b_instruct"
    "qwen2_5_72b_instruct"
    "llama_3_1_8b_instruct"
    "llama_3_1_70b_instruct"
    "meta_llama_3_8b_instruct"
)

FRAMEWORKS=("infinilm" "vllm")
if [ $# -gt 0 ]; then FRAMEWORKS=("$@"); fi

pass=0; fail=0; skip=0

for fw in "${FRAMEWORKS[@]}"; do
    section "Framework: ${fw^^}"
    for model in "${MODELS[@]}"; do
        config="${SCRIPT_DIR}/configs/infer/${fw}/${fw}_direct_${model}_b1.json"

        if [ ! -f "${config}" ]; then
            warn "${fw}/${model} — config not found"
            ((skip++)); continue
        fi

        # 检查 model_path 目录是否存在
        model_path=$(python3 -c "
import json
d = json.load(open('${config}'))
if isinstance(d, list): d = d[0]
print(d.get('config', {}).get('model_path', ''))
" 2>/dev/null)

        if [ -n "${model_path}" ] && [ ! -d "${model_path}" ]; then
            warn "${fw}/${model} — model not found: ${model_path}"
            ((skip++)); continue
        fi

        # 读出 run_id 供显示
        run_id=$(python3 -c "
import json; d=json.load(open('${config}'))
if isinstance(d,list): d=d[0]
print(d.get('run_id',''))
" 2>/dev/null)

        log="${LOG_DIR}/${fw}_${model}_b1.log"
        echo -n "  Testing ${run_id} ... "

        start=$(date +%s)
        if ${PYTHON} main.py "${config}" > "${log}" 2>&1; then
            elapsed=$(( $(date +%s) - start ))
            info "✅  (${elapsed}s)"
            ((pass++))
        else
            elapsed=$(( $(date +%s) - start ))
            error "❌  (${elapsed}s) → ${log}"
            ((fail++))
        fi
    done
done

echo ""
echo -e "${BOLD}────────────────────────────────${NC}"
echo -e "  ✅ PASS : ${pass}"
echo -e "  ❌ FAIL : ${fail}"
echo -e "  ⏭  SKIP : ${skip}"
echo -e "  Logs   : ${LOG_DIR}"
echo -e "${BOLD}────────────────────────────────${NC}"

[ "${fail}" -eq 0 ]
