<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import VChart from 'vue-echarts'
import {
  BW_TABLE,
  CI_SUMMARY,
  DIMS,
  PLATFORMS,
} from '@/data'
import { routeParamString } from '@/utils/routeParams'
import { useInfiniDashboard } from '@/composables/useInfiniDashboard'
import { useDashboardNavigation } from '@/composables/useDashboardNavigation'
import {
  computeOpRowScore,
  remarksExcludeIcFromScore,
} from '@/features/dashboard/operatorBenchmark'
import { BW_NVIDIA_BASELINE_GBPS } from '@/features/dashboard/bwBenchmark'

const route = useRoute()
const store = useInfiniDashboard()
const {
  activeDimKey,
  detailTitle,
  detailState,
  detailPlat,
  detailKpiCells,
  detailTableTab,
  tableNotice,
  scoreTabHint,
  opDetailRows,
  inferDetailTabRows,
  inferNvidiaTabRows,
  detailTestEnvLine,
  detailTestEnvSourceHint,
  trainDetailRows,
  commDetailRows,
  commNvidiaBaselineRows,
  lineChartOption,
  barChartOption,
  ciChartOption,
  ciTabKey,
  setInferTab,
} = store

const { goOverview } = useDashboardNavigation()

/** 标题与 URL 一致，避免 store 与路由瞬时不同步时误显其它平台（如访存点昇腾却显示 NVIDIA） */
const detailTitleDisplay = computed(() => {
  if (route.name !== 'detail') return detailTitle.value
  const pk = routeParamString(route.params.platKey)
  const dk = routeParamString(route.params.dimKey)
  const plat = PLATFORMS.find((p) => p.key === pk)
  const dim = DIMS.find((d) => d.key === dk)
  if (!plat || !dim) return detailTitle.value
  return `${plat.name} · ${dim.label}详情`
})

const platColor = computed(() => detailPlat.value.color)

const inferMaxTps = computed(() => {
  const rows = inferDetailTabRows.value as { tps: number }[]
  const nv = inferNvidiaTabRows.value as { tps: number }[]
  const isNvidia = detailState.value.platKey === 'nvidia'
  if (isNvidia) return rows.length ? Math.max(...rows.map((r) => r.tps), 1) : 1
  return Math.max(
    rows.length ? Math.max(...rows.map((r) => r.tps)) : 0,
    nv.length ? Math.max(...nv.map((r) => r.tps || 0)) : 0,
    1,
  )
})

type InferRow = {
  configKey: string
  model: string
  batch: number
  inLen: number
  outLen?: number
  dtype?: string
  tps: number
  ttft?: number
  decodeLatencyMs?: number
  vsNvidia?: number | null
  nvidiaBaselineTps?: number | null
}

function nvInferMatch(r: InferRow) {
  return (inferNvidiaTabRows.value as InferRow[]).find((x) => x.configKey === r.configKey)
}

type CommRowLite = { commType: string; nGpu: number; bw: number }

function nvCommMatch(r: CommRowLite) {
  return commNvidiaBaselineRows.value.find(
    (x) => x.commType === r.commType && x.nGpu === r.nGpu,
  )
}

const commBwMax = computed(() => {
  const a = commDetailRows.value.map((x) => x.bw)
  const b = commNvidiaBaselineRows.value.map((x) => x.bw)
  return Math.max(1, ...a, ...b)
})

const bwRows = computed(
  () =>
    (BW_TABLE[detailState.value.platKey as keyof typeof BW_TABLE] as
      | typeof BW_TABLE.nvidia
      | undefined) ?? [],
)

/** 访存表单元格：仅有 bw_GBps（avg）而四模式为空时，避免对 null 调用 toFixed */
function bwGbpsCell(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return '—'
  return n.toFixed(2)
}

function bwBarDenom(r: {
  add: number | null
  copy: number | null
  scale: number | null
  triad: number | null
}): number {
  return Math.max(
    r.add ?? 0,
    r.copy ?? 0,
    r.scale ?? 0,
    r.triad ?? 0,
    BW_NVIDIA_BASELINE_GBPS,
  )
}

function hasChart(opt: object) {
  return opt && Object.keys(opt).length > 0
}

function precClass(dtype: string) {
  if (dtype === 'FP16') return 'p-fp16'
  if (dtype === 'BF16') return 'p-bf16'
  return 'p-fp32'
}

function scoreCellColor(score: number) {
  return score >= 100 ? '#2e7d32' : score >= 60 ? '#e65100' : '#c62828'
}

type OpDetailRow = {
  shape: string
  dtype: string
  ic: number
  pt: number
  remarks?: string
  scoreEligible?: boolean
}

function opRowRemarks(r: OpDetailRow) {
  return String(r.remarks ?? '')
}

function opDetailScore(r: OpDetailRow): number | null {
  return computeOpRowScore(r.ic, r.pt, opRowRemarks(r))
}

function opRowWarning(r: OpDetailRow) {
  return remarksExcludeIcFromScore(opRowRemarks(r))
}

const opLatencyMax = computed(() => {
  const rows = opDetailRows.value as OpDetailRow[]
  const m = Math.max(1e-9, ...rows.flatMap((x) => [x.ic, x.pt]))
  return m
})
</script>

<template>
  <div class="detail-panel">
    <header class="detail-panel__header">
      <div class="detail-title-row">
        <div class="detail-title">{{ detailTitleDisplay }}</div>
        <a-button type="primary" @click="goOverview">返回概览</a-button>
      </div>
    </header>
    <div class="detail-panel__scroll">
      <div class="kpi-grid">
        <div
          v-for="(cell, idx) in detailKpiCells"
          :key="idx"
          class="kpi-card"
        >
          <div
            class="kpi-val"
            :style="{
              color: (cell as { muted?: boolean }).muted ? '#999' : undefined,
            }"
          >
            {{ cell.val }}
          </div>
          <div class="kpi-lbl">{{ cell.lbl }}</div>
          <div class="kpi-sub">{{ cell.sub }}</div>
        </div>
      </div>

      <!-- 测试环境 + 数据明细：用 flex gap 分隔，避免 margin 透出右侧栏灰底 -->
      <div class="detail-env-table-gap">
        <!-- 测试环境横条（各维度：n_gpu · date · device，缺省项不展示） -->
        <div
          style="
            padding: 10px 20px;
            background: #ffffff;
            border-radius: 4px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 12px;
            color: #888;
          "
        >
          <span style="font-size: 14px">🖥</span>
          <span style="font-weight: 600; color: #555">测试环境</span>
          <span style="color: #ddd">|</span>
          <span style="color: #667eea; font-weight: 500">{{ detailTestEnvLine }}</span>
          <span style="margin-left: auto; font-size: 11px; color: #bbb; font-style: italic">
            {{ detailTestEnvSourceHint }}
          </span>
        </div>

        <!-- 详情内算子筛选在 HTML 中为 display:none，此处不渲染 -->

        <div class="table-card">
      <div class="table-head-row">
        <div class="table-title">数据明细</div>
        <div class="view-toggle">
          <button
            type="button"
            class="vt-btn"
            :class="{ active: detailTableTab === 'data' }"
            @click="detailTableTab = 'data'"
          >
            数据明细
          </button>
          <button
            type="button"
            class="vt-btn"
            :class="{ active: detailTableTab === 'score' }"
            @click="detailTableTab = 'score'"
          >
            得分说明
          </button>
        </div>
      </div>
      <div
        style="
          margin-bottom: 12px;
          padding: 10px 14px;
          background: #e8f0fe;
          border-left: 4px solid #667eea;
          border-radius: 0 4px 4px 0;
          font-size: 12px;
          color: #3d5afe;
        "
      >
        {{ detailTableTab === 'data' ? tableNotice : scoreTabHint }}
      </div>

      <!-- 数据明细 -->
      <div v-show="detailTableTab === 'data'" class="table-wrap">
        <!-- 算子 -->
        <template v-if="activeDimKey === 'op'">
          <table v-if="opDetailRows.length">
            <thead>
              <tr>
                <th>Shape 配置</th>
                <th>精度</th>
                <th>InfiniCore ✦</th>
                <th>PyTorch</th>
                <th>得分</th>
                <th>备注</th>
                <th>延迟对比</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="(r, ri) in opDetailRows"
                :key="ri"
                class="op-detail-row"
                :class="{ 'op-detail-row--warn': opRowWarning(r as OpDetailRow) }"
              >
                <td class="shape-cell">{{ r.shape }}</td>
                <td>
                  <span class="prec-badge" :class="precClass(r.dtype)">{{ r.dtype }}</span>
                </td>
                <td :style="{ color: platColor, fontWeight: 600 }">{{ r.ic.toFixed(4) }}ms</td>
                <td style="color: #888">{{ r.pt.toFixed(4) }}ms</td>
                <td class="score-cell">
                  <template v-if="opDetailScore(r as OpDetailRow) != null">
                    <span
                      class="score-num"
                      :style="{ color: scoreCellColor(Math.round(opDetailScore(r as OpDetailRow)!)) }"
                    >
                      {{ Math.round(opDetailScore(r as OpDetailRow)!) }}
                    </span>
                    <span
                      :class="
                        Math.round(opDetailScore(r as OpDetailRow)!) >= 100 ? 'score-up' : 'score-dn'
                      "
                    >
                      {{ Math.round(opDetailScore(r as OpDetailRow)!) >= 100 ? '↑' : '↓' }}
                    </span>
                  </template>
                  <template v-else>
                    <span class="score-num score-na">—</span>
                    <span v-if="opRowWarning(r as OpDetailRow)" class="op-warn-icon" title="该行 InfiniCore 延迟不参与计分">⚠</span>
                  </template>
                </td>
                <td class="remarks-cell" :title="opRowRemarks(r as OpDetailRow)">
                  {{ opRowRemarks(r as OpDetailRow) || '—' }}
                </td>
                <td>
                  <div class="dual-bar">
                    <div class="dual-row">
                      <span class="dual-lbl" :style="{ color: platColor }">自研</span>
                      <div class="dual-track">
                        <div
                          class="dual-fill"
                          :style="{
                            width:
                              Math.round(((r.ic as number) / opLatencyMax) * 100) + '%',
                            background: platColor,
                          }"
                        />
                      </div>
                      <span class="dual-ms" :style="{ color: platColor }">{{ r.ic.toFixed(4) }}ms</span>
                    </div>
                    <div class="dual-row">
                      <span class="dual-lbl" style="color: #aaa">开源</span>
                      <div class="dual-track">
                        <div
                          class="dual-fill"
                          :style="{
                            width:
                              Math.round(((r.pt as number) / opLatencyMax) * 100) + '%',
                            background: '#aaa',
                          }"
                        />
                      </div>
                      <span class="dual-ms" style="color: #aaa">{{ r.pt.toFixed(4) }}ms</span>
                    </div>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
          <p v-else style="text-align: center; padding: 30px; color: #aaa">暂无该条件数据</p>
        </template>

        <!-- 推理 -->
        <template v-else-if="activeDimKey === 'infer'">
          <div style="display: flex; gap: 6px; margin-bottom: 14px">
            <button
              type="button"
              class="fpill"
              :class="{ active: detailState.inferTab === 'prefill' }"
              @click="setInferTab('prefill')"
            >
              Prefill 吞吐量
            </button>
            <button
              type="button"
              class="fpill"
              :class="{ active: detailState.inferTab === 'decode' }"
              @click="setInferTab('decode')"
            >
              Decode 吞吐量
            </button>
          </div>
          <table v-if="inferDetailTabRows.length">
            <thead>
              <tr>
                <th>模型</th>
                <th>Batch</th>
                <th>In-len</th>
                <th>Out-len</th>
                <th>精度</th>
                <th>TPS</th>
                <th v-if="detailState.inferTab === 'prefill'">TTFT</th>
                <th v-if="detailState.inferTab === 'decode'">Decode 延迟</th>
                <th v-if="detailState.platKey !== 'nvidia'">vs NVIDIA</th>
                <th>对比</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="r in inferDetailTabRows as InferRow[]" :key="r.configKey">
                <td style="font-family: monospace; font-size: 12px" :style="{ color: platColor }">
                  {{ r.model }}
                </td>
                <td>{{ r.batch }}</td>
                <td>{{ r.inLen }}</td>
                <td>{{ r.outLen ?? '—' }}</td>
                <td>
                  <span v-if="r.dtype" class="prec-badge" :class="precClass(r.dtype)">{{ r.dtype }}</span>
                  <span v-else style="color: #ccc">—</span>
                </td>
                <td style="font-weight: 700" :style="{ color: platColor }">
                  {{ r.tps.toLocaleString() }}
                </td>
                <td v-if="detailState.inferTab === 'prefill'" style="color: #888">
                  {{ r.ttft ? r.ttft + 'ms' : '—' }}
                </td>
                <td v-if="detailState.inferTab === 'decode'" style="color: #888">
                  {{ r.decodeLatencyMs != null ? r.decodeLatencyMs + 'ms' : '—' }}
                </td>
                <td
                  v-if="detailState.platKey !== 'nvidia'"
                  :style="{
                    color:
                      r.vsNvidia != null
                        ? r.vsNvidia >= 100
                          ? '#2e7d32'
                          : '#e65100'
                        : '#999',
                    fontWeight: 700,
                  }"
                >
                  {{ r.vsNvidia != null ? r.vsNvidia + '%' : '—' }}
                </td>
                <td>
                  <div v-if="detailState.platKey === 'nvidia'" class="dual-bar">
                    <div class="dual-row">
                      <div class="dual-track" style="flex: 1">
                        <div
                          class="dual-fill"
                          :style="{
                            width: Math.round((r.tps / inferMaxTps) * 100) + '%',
                            background: platColor,
                          }"
                        />
                      </div>
                      <span class="dual-ms" :style="{ color: platColor }">{{ r.tps.toLocaleString() }}</span>
                    </div>
                  </div>
                  <div v-else class="dual-bar">
                    <div class="dual-row">
                      <span class="dual-lbl" :style="{ color: platColor }">{{ detailPlat.logo }}</span>
                      <div class="dual-track">
                        <div
                          class="dual-fill"
                          :style="{
                            width: Math.round((r.tps / inferMaxTps) * 100) + '%',
                            background: platColor,
                          }"
                        />
                      </div>
                      <span class="dual-ms" :style="{ color: platColor }">{{ r.tps.toLocaleString() }}</span>
                    </div>
                    <div v-if="nvInferMatch(r)" class="dual-row">
                      <span class="dual-lbl" style="color: #aaa">NV</span>
                      <div class="dual-track">
                        <div
                          class="dual-fill"
                          :style="{
                            width:
                              Math.round(
                                ((nvInferMatch(r)!.tps || 0) / inferMaxTps) * 100,
                              ) + '%',
                            background: '#aaa',
                          }"
                        />
                      </div>
                      <span class="dual-ms" style="color: #aaa">{{
                        (nvInferMatch(r)!.tps ?? 0).toLocaleString()
                      }}</span>
                    </div>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
          <p v-else style="text-align: center; padding: 30px; color: #aaa">暂无该平台数据</p>
        </template>

        <!-- 训练 -->
        <template v-else-if="activeDimKey === 'train'">
          <table v-if="trainDetailRows.length">
            <thead>
              <tr>
                <th>框架</th>
                <th>模型</th>
                <th>并行配置</th>
                <th>精度</th>
                <th>Flash Attn</th>
                <th>吞吐</th>
                <template v-if="detailState.platKey !== 'nvidia'">
                  <th>NVIDIA 基线</th>
                  <th>vs NVIDIA</th>
                </template>
                <th>备注</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(r, ri) in trainDetailRows" :key="ri">
                <td style="font-weight: 600">{{ r.framework }}</td>
                <td style="font-family: monospace; font-size: 12px" :style="{ color: platColor }">
                  {{ r.model }}
                </td>
                <td style="font-size: 12px">{{ r.parallel }}</td>
                <td><span class="prec-badge" :class="precClass(r.dtype)">{{ r.dtype }}</span></td>
                <td style="font-size: 12px">
                  <span v-if="r.flashAttn === 'on'" style="color: #2e7d32">✓ on</span>
                  <span v-else style="color: #aaa">{{ r.flashAttn }}</span>
                </td>
                <td style="font-weight: 700" :style="{ color: platColor }">
                  {{ r.tps.toLocaleString() }} tpps
                </td>
                <template v-if="detailState.platKey !== 'nvidia'">
                  <td style="color: #888">{{ r.baseline.toLocaleString() }} tpps</td>
                  <td
                    :style="{
                      fontWeight: 700,
                      color:
                        r.vsA100 >= 100 ? '#2e7d32' : r.vsA100 >= 70 ? '#e65100' : '#c62828',
                    }"
                  >
                    {{ r.vsA100 }}%
                  </td>
                </template>
                <td style="font-size: 11px; color: #aaa">{{ r.note || '' }}</td>
              </tr>
            </tbody>
          </table>
          <p v-else style="text-align: center; padding: 30px; color: #aaa">暂无该平台数据</p>
        </template>

        <!-- 通信 -->
        <template v-else-if="activeDimKey === 'comm'">
          <table v-if="commDetailRows.length">
            <thead>
              <tr>
                <th>Link 类型</th>
                <th>通信类型</th>
                <th>GPU 数</th>
                <th>带宽</th>
                <template v-if="detailState.platKey !== 'nvidia'">
                  <th>NVIDIA 基线</th>
                  <th>vs NVIDIA</th>
                </template>
                <th>带宽对比</th>
                <th>备注</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="r in commDetailRows" :key="r.commType + '-' + r.nGpu">
                <td style="font-weight: 600" :style="{ color: platColor }">{{ r.linkType }}</td>
                <td><span class="prec-badge p-fp16">{{ r.commType }}</span></td>
                <td>{{ r.nGpu }} GPU</td>
                <td style="font-weight: 700" :style="{ color: platColor }">{{ r.bw }} GB/s</td>
                <td v-if="detailState.platKey !== 'nvidia'" style="color: #888">{{ r.baseline }} GB/s</td>
                <td
                  v-if="detailState.platKey !== 'nvidia'"
                  :style="{
                    fontWeight: 700,
                    color:
                      r.vsA100 >= 100 ? '#2e7d32' : r.vsA100 >= 70 ? '#e65100' : '#c62828',
                  }"
                >
                  {{ r.vsA100 }}%
                </td>
                <td>
                  <div v-if="detailState.platKey === 'nvidia'" class="dual-bar">
                    <div class="dual-row">
                      <div class="dual-track" style="flex: 1">
                        <div
                          class="dual-fill"
                          :style="{
                            width:
                              Math.round(
                                (r.bw / commBwMax) *
                                  100,
                              ) + '%',
                            background: platColor,
                          }"
                        />
                      </div>
                      <span class="dual-ms" :style="{ color: platColor }">{{ r.bw }}</span>
                    </div>
                  </div>
                  <div v-else class="dual-bar">
                    <div class="dual-row">
                      <span class="dual-lbl" :style="{ color: platColor }">{{ detailPlat.logo }}</span>
                      <div class="dual-track">
                        <div
                          class="dual-fill"
                          :style="{
                            width:
                              Math.round(
                                (r.bw / commBwMax) *
                                  100,
                              ) + '%',
                            background: platColor,
                          }"
                        />
                      </div>
                      <span class="dual-ms" :style="{ color: platColor }">{{ r.bw }}</span>
                    </div>
                    <div v-if="nvCommMatch(r)" class="dual-row">
                      <span class="dual-lbl" style="color: #aaa">NV</span>
                      <div class="dual-track">
                        <div
                          class="dual-fill"
                          :style="{
                            width:
                              Math.round(
                                ((nvCommMatch(r)!.bw || 0) / commBwMax) *
                                  100,
                              ) + '%',
                            background: '#aaa',
                          }"
                        />
                      </div>
                      <span class="dual-ms" style="color: #aaa">{{ nvCommMatch(r)!.bw }}</span>
                    </div>
                  </div>
                </td>
                <td style="font-size: 11px; color: #aaa">{{ r.note || '' }}</td>
              </tr>
            </tbody>
          </table>
          <p v-else style="text-align: center; padding: 30px; color: #aaa">暂无该平台数据</p>
        </template>

        <!-- 访存 -->
        <template v-else-if="activeDimKey === 'bw'">
          <table v-if="bwRows.length">
            <thead>
              <tr>
                <th>型号</th>
                <th>add GB/s</th>
                <th>copy GB/s</th>
                <th>scale GB/s</th>
                <th>triad GB/s</th>
                <th>均值 GB/s</th>
                <th>vs NVIDIA</th>
                <th>带宽对比</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(r, ri) in bwRows" :key="ri">
                <template v-if="r.avg == null">
                  <td :style="{ color: platColor, fontWeight: 600 }">{{ r.model }}</td>
                  <td colspan="7" style="color: #aaa; font-style: italic">数据待补充</td>
                </template>
                <template v-else>
                  <td :style="{ color: platColor, fontWeight: 600 }">{{ r.model }}</td>
                  <td style="font-weight: 600">{{ bwGbpsCell(r.add) }}</td>
                  <td>{{ bwGbpsCell(r.copy) }}</td>
                  <td>{{ bwGbpsCell(r.scale) }}</td>
                  <td>{{ bwGbpsCell(r.triad) }}</td>
                  <td style="font-weight: 700" :style="{ color: platColor }">{{ bwGbpsCell(r.avg) }}</td>
                  <td
                    :style="{
                      fontWeight: 700,
                      color: r.vsNvidia >= 100 ? '#2e7d32' : '#e65100',
                    }"
                  >
                    {{ r.vsNvidia }}%
                  </td>
                  <td>
                    <div class="dual-bar">
                      <div class="dual-row">
                        <span class="dual-lbl" :style="{ color: platColor }">{{ detailPlat.logo }}</span>
                        <div class="dual-track">
                          <div
                            class="dual-fill"
                            :style="{
                              width:
                                Math.round(
                                  (((r.avg ?? 0) / bwBarDenom(r)) || 0) * 100,
                                ) + '%',
                              background: platColor,
                            }"
                          />
                        </div>
                        <span class="dual-ms" :style="{ color: platColor }">{{
                          r.avg != null && Number.isFinite(r.avg) ? Math.round(r.avg).toString() : '—'
                        }}</span>
                      </div>
                      <div class="dual-row">
                        <span class="dual-lbl" style="color: #aaa">NVIDIA 基线</span>
                        <div class="dual-track">
                          <div
                            class="dual-fill"
                            :style="{
                              width:
                                Math.round(
                                  ((BW_NVIDIA_BASELINE_GBPS / bwBarDenom(r)) || 0) * 100,
                                ) + '%',
                              background: '#aaa',
                            }"
                          />
                        </div>
                        <span class="dual-ms" style="color: #aaa">{{
                          BW_NVIDIA_BASELINE_GBPS.toFixed(1)
                        }}</span>
                      </div>
                    </div>
                  </td>
                </template>
              </tr>
            </tbody>
          </table>
          <p v-else style="text-align: center; padding: 30px; color: #aaa">暂无该平台数据</p>
        </template>
      </div>

      <!-- 得分说明 Tab：仅文案 -->
      <div v-show="detailTableTab === 'score'" class="table-wrap">
        <p style="text-align: center; padding: 30px; color: #aaa; font-size: 13px">
          {{ scoreTabHint }}
        </p>
      </div>
        </div>
      </div>

    <!-- 主图（与 HTML 标题文案一致） -->
    <div class="charts-grid">
      <div class="chart-card">
        <div class="chart-title">延迟趋势对比</div>
        <v-chart
          v-if="hasChart(lineChartOption)"
          class="detail-chart"
          :option="lineChartOption"
          autoresize
        />
      </div>
      <div class="chart-card">
        <div class="chart-title">各算子平均得分</div>
        <v-chart
          v-if="hasChart(barChartOption)"
          class="detail-chart"
          :option="barChartOption"
          autoresize
        />
      </div>
    </div>

    <!-- CI -->
    <div class="ci-card">
      <a-tabs v-model:activeKey="ciTabKey">
        <a-tab-pane key="ci-stats" tab="CI 运行统计">
          <div class="ci-stats-grid">
            <div class="ci-stat">
              <div class="ci-stat-val">{{ CI_SUMMARY.runCount }}</div>
              <div class="ci-stat-lbl">CI 运行次数</div>
            </div>
            <div class="ci-stat">
              <div class="ci-stat-val" style="color: #2e7d32">{{ CI_SUMMARY.avgSuccessRate }}</div>
              <div class="ci-stat-lbl">平均成功率</div>
            </div>
            <div class="ci-stat">
              <div class="ci-stat-val" style="color: #2e7d32">{{ CI_SUMMARY.last10SuccessRate }}</div>
              <div class="ci-stat-lbl">最近10次成功率</div>
            </div>
            <div class="ci-stat">
              <div class="ci-stat-val" style="color: #ef5350">{{ CI_SUMMARY.failureCount }}</div>
              <div class="ci-stat-lbl">失败用例</div>
            </div>
          </div>
          <v-chart v-if="hasChart(ciChartOption)" class="ci-chart" :option="ciChartOption" autoresize />
        </a-tab-pane>
        <a-tab-pane key="dispatcher" tab="Dispatcher 汇总">
          <p style="text-align: center; padding: 30px; color: #aaa; font-size: 13px">
            Dispatcher 汇总记录 — 接入真实 CI 数据后展示
          </p>
        </a-tab-pane>
        <a-tab-pane key="ci-detail" tab="CI 详细记录">
          <p style="text-align: center; padding: 30px; color: #aaa; font-size: 13px">
            CI 详细记录 — 可跳转至独立页面查看
          </p>
        </a-tab-pane>
        <a-tab-pane key="failure" tab="失败详情">
          <p style="text-align: center; padding: 30px; color: #aaa; font-size: 13px">
            失败详情 — 可跳转至独立页面查看
          </p>
        </a-tab-pane>
      </a-tabs>
    </div>
    </div>
  </div>
</template>

<style scoped>
.detail-panel {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
  height: 100%;
  /* 与外层白卡片一致，避免块间距透出右侧栏灰底 */
  background-color: #fff;
}
.detail-panel__header {
  flex-shrink: 0;
}
.detail-panel__scroll {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  overflow-x: hidden;
  padding-bottom: 4px;
  background-color: #fff;
}
.detail-panel__scroll .kpi-grid {
  margin-bottom: 16px;
}
.detail-env-table-gap {
  display: flex;
  flex-direction: column;
  gap: 16px;
  background-color: #ffffff;
}
.detail-chart {
  height: 280px;
  width: 100%;
}
.ci-chart {
  height: 160px;
  width: 100%;
}
</style>
