<script setup lang="ts">
import { computed, h } from 'vue'
import type { ColumnsType } from 'ant-design-vue/es/table'
import type { TrainTableRow } from '@/features/dashboard/trainBenchmark'
import type { CommImportRow } from '@/features/dashboard/commBenchmark'
import type { BwDetailRow } from '@/features/dashboard/bwBenchmark'
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
  tableNotice,
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

/** 列宽总和与 scroll.x 一致，tableLayout=fixed 下各列更均匀 */
const opTableColumns: ColumnsType<OpDetailRow> = [
  { title: 'Shape 配置', dataIndex: 'shape', key: 'shape', width: 152, ellipsis: true },
  { title: '精度', key: 'dtype', width: 84 },
  { title: 'InfiniCore ✦', key: 'ic', width: 118 },
  { title: 'PyTorch', key: 'pt', width: 118 },
  { title: '得分', key: 'score', width: 92 },
  { title: '延迟对比', key: 'dual', width: 268, minWidth: 240 },
]

function opRowKey(r: OpDetailRow, i?: number) {
  return `${i ?? 0}-${r.shape}-${r.dtype}`
}

function opCustomRow(r: OpDetailRow) {
  return {
    class: opRowWarning(r) ? 'op-detail-row op-detail-row--warn' : 'op-detail-row',
  }
}

const inferTableColumns = computed<ColumnsType<InferRow>>(() => {
  const cols: ColumnsType<InferRow> = [
    { title: '模型', key: 'model', minWidth: 160, width: 168 },
    { title: 'Batch', dataIndex: 'batch', key: 'batch', width: 84 },
    { title: 'In-len', dataIndex: 'inLen', key: 'inLen', width: 84 },
    { title: 'Out-len', key: 'outLen', width: 84 },
    { title: '精度', key: 'dtype', width: 88 },
    { title: 'TPS', key: 'tps', width: 108 },
  ]
  if (detailState.value.inferTab === 'prefill') {
    cols.push({ title: 'TTFT', key: 'ttft', width: 100 })
  } else {
    cols.push({ title: 'Decode 延迟', key: 'decodeLatency', width: 118 })
  }
  if (detailState.value.platKey !== 'nvidia') {
    cols.push({ title: 'vs NVIDIA', key: 'vsNvidia', width: 100 })
  }
  cols.push({ title: '对比', key: 'compare', minWidth: 200, width: 220 })
  return cols
})

const trainTableColumns = computed<ColumnsType<TrainTableRow>>(() => {
  const cols: ColumnsType<TrainTableRow> = [
    { title: '框架', key: 'framework', width: 104 },
    { title: '模型', key: 'model', minWidth: 132, width: 140 },
    { title: '并行配置', key: 'parallel', width: 132 },
    { title: '精度', key: 'dtype', width: 88 },
    { title: 'Flash Attn', key: 'flashAttn', width: 116 },
    { title: '吞吐', key: 'tps', width: 132 },
  ]
  if (detailState.value.platKey !== 'nvidia') {
    cols.push(
      { title: 'NVIDIA 基线', key: 'baseline', width: 124 },
      { title: 'vs NVIDIA', key: 'vsA100', width: 104 },
    )
  }
  return cols
})

const commTableColumns = computed<ColumnsType<CommImportRow>>(() => {
  const cols: ColumnsType<CommImportRow> = [
    { title: 'Link 类型', key: 'linkType', width: 112 },
    { title: '通信类型', key: 'commType', width: 112 },
    { title: 'GPU 数', key: 'nGpu', width: 92 },
    { title: '带宽', key: 'bw', width: 104 },
  ]
  if (detailState.value.platKey !== 'nvidia') {
    cols.push(
      { title: 'NVIDIA 基线', key: 'baseline', width: 116 },
      { title: 'vs NVIDIA', key: 'vsA100', width: 104 },
    )
  }
  cols.push({ title: '带宽对比', key: 'compare', minWidth: 200, width: 228 })
  return cols
})

const bwTableColumns = computed<ColumnsType<BwDetailRow>>(() => {
  const pc = platColor.value
  const logo = detailPlat.value.logo
  const pendingCell = (r: BwDetailRow) => (r.avg == null ? { colSpan: 0 } : {})

  return [
    {
      title: '型号',
      key: 'model',
      dataIndex: 'model',
      width: 120,
      customRender: ({ record }) =>
        h('span', { style: { color: pc, fontWeight: 600 } }, record.model),
    },
    {
      title: 'add GB/s',
      key: 'add',
      width: 110,
      customCell: (r: BwDetailRow) => (r.avg == null ? { colSpan: 7 } : {}),
      customRender: ({ record }) =>
        record.avg == null
          ? h('span', { style: { color: '#aaa', fontStyle: 'italic' } }, '数据待补充')
          : h('span', { style: { fontWeight: 600 } }, bwGbpsCell(record.add)),
    },
    {
      title: 'copy GB/s',
      key: 'copy',
      customCell: pendingCell,
      customRender: ({ record }) =>
        record.avg == null ? null : bwGbpsCell(record.copy),
    },
    {
      title: 'scale GB/s',
      key: 'scale',
      customCell: pendingCell,
      customRender: ({ record }) =>
        record.avg == null ? null : bwGbpsCell(record.scale),
    },
    {
      title: 'triad GB/s',
      key: 'triad',
      customCell: pendingCell,
      customRender: ({ record }) =>
        record.avg == null ? null : bwGbpsCell(record.triad),
    },
    {
      title: '均值 GB/s',
      key: 'avg',
      customCell: pendingCell,
      customRender: ({ record }) =>
        record.avg == null
          ? null
          : h('span', { style: { fontWeight: 700, color: pc } }, bwGbpsCell(record.avg)),
    },
    {
      title: 'vs NVIDIA',
      key: 'vsNvidia',
      width: 110,
      customCell: pendingCell,
      customRender: ({ record }) =>
        record.avg == null
          ? null
          : h(
              'span',
              {
                style: {
                  fontWeight: 700,
                  color: record.vsNvidia >= 100 ? '#2e7d32' : '#e65100',
                },
              },
              `${record.vsNvidia}%`,
            ),
    },
    {
      title: '带宽对比',
      key: 'dual',
      width: 260,
      customCell: pendingCell,
      customRender: ({ record }) => {
        if (record.avg == null) return null
        const denom = bwBarDenom(record)
        const avgW =
          Math.round((((record.avg ?? 0) / denom) || 0) * 100) + '%'
        const nvW =
          Math.round(((BW_NVIDIA_BASELINE_GBPS / denom) || 0) * 100) + '%'
        const avgMs =
          record.avg != null && Number.isFinite(record.avg)
            ? Math.round(record.avg).toString()
            : '—'
        return h('div', { class: 'dual-bar' }, [
          h('div', { class: 'dual-row' }, [
            h('span', { class: 'dual-lbl dual-lbl--bw', style: { color: pc } }, logo),
            h('div', { class: 'dual-track' }, [
              h('div', {
                class: 'dual-fill',
                style: { width: avgW, background: pc },
              }),
            ]),
            h('span', { class: 'dual-ms', style: { color: pc } }, avgMs),
          ]),
          h('div', { class: 'dual-row' }, [
            h('span', { class: 'dual-lbl dual-lbl--bw', style: { color: '#aaa' } }, 'NVIDIA 基线'),
            h('div', { class: 'dual-track' }, [
              h('div', {
                class: 'dual-fill',
                style: { width: nvW, background: '#aaa' },
              }),
            ]),
            h(
              'span',
              { class: 'dual-ms', style: { color: '#aaa' } },
              BW_NVIDIA_BASELINE_GBPS.toFixed(1),
            ),
          ]),
        ])
      },
    },
  ]
})

function bwRowKey(r: BwDetailRow, i?: number) {
  return `${i ?? 0}-${r.model}`
}

function trainRowKey(r: TrainTableRow, i?: number) {
  return `${r.matchKey}-${i ?? 0}`
}

function commRowKey(r: CommImportRow) {
  return `${r.commType}-${r.nGpu}`
}

function inferRowKey(r: InferRow) {
  return r.configKey
}
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
      <!-- 测试环境横条（各维度：n_gpu · date · device，缺省项不展示）：紧接标题下方、KPI 卡片上方 -->
      <div class="detail-test-env-bar">
        <span class="detail-test-env-bar__icon" aria-hidden="true">🖥</span>
        <span class="detail-test-env-bar__title">测试环境</span>
        <span class="detail-test-env-bar__sep">|</span>
        <span class="detail-test-env-bar__line">{{ detailTestEnvLine }}</span>
        <span class="detail-test-env-bar__source">{{ detailTestEnvSourceHint }}</span>
      </div>

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

      <!-- 数据明细：用 flex gap 分隔，避免 margin 透出右侧栏灰底 -->
      <div class="detail-env-table-gap">
        <!-- 详情内算子筛选在 HTML 中为 display:none，此处不渲染 -->

        <div class="table-card">
      <div class="table-head-row">
        <div class="table-title">数据明细</div>
      </div>
      <div class="detail-table-notice">
        <div class="detail-table-notice__label">得分说明：</div>
        <div class="detail-table-notice__body">{{ tableNotice }}</div>
      </div>

      <!-- 数据明细 -->
      <div class="table-wrap">
        <!-- 算子 -->
        <template v-if="activeDimKey === 'op'">
          <a-table
            v-if="opDetailRows.length"
            :columns="opTableColumns"
            :data-source="(opDetailRows as OpDetailRow[])"
            :pagination="false"
            :bordered="false"
            table-layout="fixed"
            :scroll="{ x: 832 }"
            :row-key="opRowKey"
            :custom-row="opCustomRow"
          >
            <template #bodyCell="{ column, record }">
              <template v-if="column.key === 'shape'">
                <span class="shape-cell">{{ record.shape }}</span>
              </template>
              <template v-else-if="column.key === 'dtype'">
                <span class="prec-badge" :class="precClass(record.dtype)">{{ record.dtype }}</span>
              </template>
              <template v-else-if="column.key === 'ic'">
                <span :style="{ color: platColor, fontWeight: 600 }">{{ record.ic.toFixed(4) }}ms</span>
              </template>
              <template v-else-if="column.key === 'pt'">
                <span style="color: #888">{{ record.pt.toFixed(4) }}ms</span>
              </template>
              <template v-else-if="column.key === 'score'">
                <div class="score-cell">
                  <template v-if="opDetailScore(record as OpDetailRow) != null">
                    <span
                      class="score-num"
                      :style="{
                        color: scoreCellColor(Math.round(opDetailScore(record as OpDetailRow)!)),
                      }"
                    >
                      {{ Math.round(opDetailScore(record as OpDetailRow)!) }}
                    </span>
                    <span
                      :class="
                        Math.round(opDetailScore(record as OpDetailRow)!) >= 100
                          ? 'score-up'
                          : 'score-dn'
                      "
                    >
                      {{ Math.round(opDetailScore(record as OpDetailRow)!) >= 100 ? '↑' : '↓' }}
                    </span>
                  </template>
                  <template v-else>
                    <span class="score-num score-na">—</span>
                    <span
                      v-if="opRowWarning(record as OpDetailRow)"
                      class="op-warn-icon"
                      title="该行 InfiniCore 延迟不参与计分"
                      >⚠</span
                    >
                  </template>
                </div>
              </template>
              <template v-else-if="column.key === 'dual'">
                <div class="dual-bar">
                  <div class="dual-row">
                    <span class="dual-lbl" :style="{ color: platColor }">自研</span>
                    <div class="dual-track">
                      <div
                        class="dual-fill"
                        :style="{
                          width: Math.round(((record.ic as number) / opLatencyMax) * 100) + '%',
                          background: platColor,
                        }"
                      />
                    </div>
                    <span class="dual-ms" :style="{ color: platColor }">{{ record.ic.toFixed(4) }}ms</span>
                  </div>
                  <div class="dual-row">
                    <span class="dual-lbl" style="color: #aaa">开源</span>
                    <div class="dual-track">
                      <div
                        class="dual-fill"
                        :style="{
                          width: Math.round(((record.pt as number) / opLatencyMax) * 100) + '%',
                          background: '#aaa',
                        }"
                      />
                    </div>
                    <span class="dual-ms" style="color: #aaa">{{ record.pt.toFixed(4) }}ms</span>
                  </div>
                </div>
              </template>
            </template>
          </a-table>
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
          <a-table
            v-if="inferDetailTabRows.length"
            :columns="inferTableColumns"
            :data-source="(inferDetailTabRows as InferRow[])"
            :pagination="false"
            :bordered="false"
            table-layout="fixed"
            :scroll="{ x: 1080 }"
            :row-key="inferRowKey"
          >
            <template #bodyCell="{ column, record, text }">
              <template v-if="column.key === 'model'">
                <span style="font-family: monospace" :style="{ color: platColor }">{{ record.model }}</span>
              </template>
              <template v-else-if="column.key === 'batch'">{{ text }}</template>
              <template v-else-if="column.key === 'inLen'">{{ text }}</template>
              <template v-else-if="column.key === 'outLen'">
                {{ record.outLen ?? '—' }}
              </template>
              <template v-else-if="column.key === 'dtype'">
                <span v-if="record.dtype" class="prec-badge" :class="precClass(record.dtype)">{{
                  record.dtype
                }}</span>
                <span v-else style="color: #ccc">—</span>
              </template>
              <template v-else-if="column.key === 'tps'">
                <span style="font-weight: 700" :style="{ color: platColor }">{{
                  record.tps.toLocaleString()
                }}</span>
              </template>
              <template v-else-if="column.key === 'ttft'">
                <span style="color: #888">{{ record.ttft ? record.ttft + 'ms' : '—' }}</span>
              </template>
              <template v-else-if="column.key === 'decodeLatency'">
                <span style="color: #888">{{
                  record.decodeLatencyMs != null ? record.decodeLatencyMs + 'ms' : '—'
                }}</span>
              </template>
              <template v-else-if="column.key === 'vsNvidia'">
                <span
                  :style="{
                    color:
                      record.vsNvidia != null
                        ? record.vsNvidia >= 100
                          ? '#2e7d32'
                          : '#e65100'
                        : '#999',
                    fontWeight: 700,
                  }"
                >
                  {{ record.vsNvidia != null ? record.vsNvidia + '%' : '—' }}
                </span>
              </template>
              <template v-else-if="column.key === 'compare'">
                <div v-if="detailState.platKey === 'nvidia'" class="dual-bar">
                  <div class="dual-row">
                    <div class="dual-track" style="flex: 1">
                      <div
                        class="dual-fill"
                        :style="{
                          width: Math.round((record.tps / inferMaxTps) * 100) + '%',
                          background: platColor,
                        }"
                      />
                    </div>
                    <span class="dual-ms" :style="{ color: platColor }">{{
                      record.tps.toLocaleString()
                    }}</span>
                  </div>
                </div>
                <div v-else class="dual-bar">
                  <div class="dual-row">
                    <span class="dual-lbl" :style="{ color: platColor }">{{ detailPlat.logo }}</span>
                    <div class="dual-track">
                      <div
                        class="dual-fill"
                        :style="{
                          width: Math.round((record.tps / inferMaxTps) * 100) + '%',
                          background: platColor,
                        }"
                      />
                    </div>
                    <span class="dual-ms" :style="{ color: platColor }">{{
                      record.tps.toLocaleString()
                    }}</span>
                  </div>
                  <div v-if="nvInferMatch(record as InferRow)" class="dual-row">
                    <span class="dual-lbl" style="color: #aaa">NV</span>
                    <div class="dual-track">
                      <div
                        class="dual-fill"
                        :style="{
                          width:
                            Math.round(
                              ((nvInferMatch(record as InferRow)!.tps || 0) / inferMaxTps) * 100,
                            ) + '%',
                          background: '#aaa',
                        }"
                      />
                    </div>
                    <span class="dual-ms" style="color: #aaa">{{
                      (nvInferMatch(record as InferRow)!.tps ?? 0).toLocaleString()
                    }}</span>
                  </div>
                </div>
              </template>
              <template v-else>{{ text }}</template>
            </template>
          </a-table>
          <p v-else style="text-align: center; padding: 30px; color: #aaa">暂无该平台数据</p>
        </template>

        <!-- 训练 -->
        <template v-else-if="activeDimKey === 'train'">
          <a-table
            v-if="trainDetailRows.length"
            :columns="trainTableColumns"
            :data-source="trainDetailRows"
            :pagination="false"
            :bordered="false"
            table-layout="fixed"
            :scroll="{ x: 1040 }"
            :row-key="trainRowKey"
          >
            <template #bodyCell="{ column, record }">
              <template v-if="column.key === 'framework'">
                <span style="font-weight: 600">{{ record.framework }}</span>
              </template>
              <template v-else-if="column.key === 'model'">
                <span style="font-family: monospace" :style="{ color: platColor }">{{ record.model }}</span>
              </template>
              <template v-else-if="column.key === 'parallel'">
                {{ record.parallel }}
              </template>
              <template v-else-if="column.key === 'dtype'">
                <span class="prec-badge" :class="precClass(record.dtype)">{{ record.dtype }}</span>
              </template>
              <template v-else-if="column.key === 'flashAttn'">
                <span v-if="record.flashAttn === 'on'" style="color: #2e7d32">✓ on</span>
                <span v-else style="color: #aaa">{{ record.flashAttn }}</span>
              </template>
              <template v-else-if="column.key === 'tps'">
                <span style="font-weight: 700" :style="{ color: platColor }">
                  {{ record.tps.toLocaleString() }} tpps
                </span>
              </template>
              <template v-else-if="column.key === 'baseline'">
                <span style="color: #888">{{ record.baseline.toLocaleString() }} tpps</span>
              </template>
              <template v-else-if="column.key === 'vsA100'">
                <span
                  :style="{
                    fontWeight: 700,
                    color:
                      record.vsA100 >= 100
                        ? '#2e7d32'
                        : record.vsA100 >= 70
                          ? '#e65100'
                          : '#c62828',
                  }"
                >
                  {{ record.vsA100 }}%
                </span>
              </template>
            </template>
          </a-table>
          <p v-else style="text-align: center; padding: 30px; color: #aaa">暂无该平台数据</p>
        </template>

        <!-- 通信 -->
        <template v-else-if="activeDimKey === 'comm'">
          <a-table
            v-if="commDetailRows.length"
            :columns="commTableColumns"
            :data-source="commDetailRows"
            :pagination="false"
            :bordered="false"
            table-layout="fixed"
            :scroll="{ x: 880 }"
            :row-key="commRowKey"
          >
            <template #bodyCell="{ column, record }">
              <template v-if="column.key === 'linkType'">
                <span style="font-weight: 600" :style="{ color: platColor }">{{ record.linkType }}</span>
              </template>
              <template v-else-if="column.key === 'commType'">
                <span class="prec-badge p-fp16">{{ record.commType }}</span>
              </template>
              <template v-else-if="column.key === 'nGpu'">{{ record.nGpu }} GPU</template>
              <template v-else-if="column.key === 'bw'">
                <span style="font-weight: 700" :style="{ color: platColor }">{{ record.bw }} GB/s</span>
              </template>
              <template v-else-if="column.key === 'baseline'">
                <span style="color: #888">{{ record.baseline }} GB/s</span>
              </template>
              <template v-else-if="column.key === 'vsA100'">
                <span
                  :style="{
                    fontWeight: 700,
                    color:
                      record.vsA100 >= 100
                        ? '#2e7d32'
                        : record.vsA100 >= 70
                          ? '#e65100'
                          : '#c62828',
                  }"
                >
                  {{ record.vsA100 }}%
                </span>
              </template>
              <template v-else-if="column.key === 'compare'">
                <div v-if="detailState.platKey === 'nvidia'" class="dual-bar">
                  <div class="dual-row">
                    <div class="dual-track" style="flex: 1">
                      <div
                        class="dual-fill"
                        :style="{
                          width: Math.round((record.bw / commBwMax) * 100) + '%',
                          background: platColor,
                        }"
                      />
                    </div>
                    <span class="dual-ms" :style="{ color: platColor }">{{ record.bw }}</span>
                  </div>
                </div>
                <div v-else class="dual-bar">
                  <div class="dual-row">
                    <span class="dual-lbl" :style="{ color: platColor }">{{ detailPlat.logo }}</span>
                    <div class="dual-track">
                      <div
                        class="dual-fill"
                        :style="{
                          width: Math.round((record.bw / commBwMax) * 100) + '%',
                          background: platColor,
                        }"
                      />
                    </div>
                    <span class="dual-ms" :style="{ color: platColor }">{{ record.bw }}</span>
                  </div>
                  <div v-if="nvCommMatch(record as CommRowLite)" class="dual-row">
                    <span class="dual-lbl" style="color: #aaa">NV</span>
                    <div class="dual-track">
                      <div
                        class="dual-fill"
                        :style="{
                          width:
                            Math.round(
                              ((nvCommMatch(record as CommRowLite)!.bw || 0) / commBwMax) * 100,
                            ) + '%',
                          background: '#aaa',
                        }"
                      />
                    </div>
                    <span class="dual-ms" style="color: #aaa">{{
                      nvCommMatch(record as CommRowLite)!.bw
                    }}</span>
                  </div>
                </div>
              </template>
            </template>
          </a-table>
          <p v-else style="text-align: center; padding: 30px; color: #aaa">暂无该平台数据</p>
        </template>

        <!-- 访存 -->
        <template v-else-if="activeDimKey === 'bw'">
          <a-table
            v-if="bwRows.length"
            :columns="bwTableColumns"
            :data-source="bwRows"
            :pagination="false"
            :bordered="false"
            :scroll="{ x: 'max-content' }"
            :row-key="bwRowKey"
          />
          <p v-else style="text-align: center; padding: 30px; color: #aaa">暂无该平台数据</p>
        </template>
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
/* 与 Ant Design Alert type="info" 一致的提示条样式 */
.detail-panel__scroll .detail-test-env-bar {
  margin-bottom: 16px;
  padding: 10px 14px;
  background: #e6f7ff;
  border-left: 4px solid #1890ff;
  border-radius: 0;
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 14px;
  line-height: 1.5;
  color: #0958d9;
}
.detail-test-env-bar__icon {
  flex-shrink: 0;
  opacity: 0.9;
}
.detail-test-env-bar__title {
  flex-shrink: 0;
  font-weight: 600;
  color: #0958d9;
}
.detail-test-env-bar__sep {
  flex-shrink: 0;
  color: #69c0ff;
  font-weight: 300;
}
.detail-test-env-bar__line {
  flex: 1;
  min-width: 0;
  font-weight: 500;
  color: #0958d9;
}
.detail-test-env-bar__source {
  flex-shrink: 0;
  margin-left: auto;
  max-width: 42%;
  font-size: 13px;
  color: rgba(0, 0, 0, 0.45);
  font-style: italic;
  text-align: right;
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
/* 数据明细上方：左「得分说明」、右正文，两列 */
.detail-table-notice {
  display: grid;
  grid-template-columns: max-content 1fr;
  column-gap: 0;
  align-items: start;
  margin-bottom: 12px;
  font-size: 12px;
  line-height: 1.55;
  color: #8c8c8c;
}
.detail-table-notice__label {
  white-space: nowrap;
  font-weight: 500;
  color: #737373;
  margin: 0;
  padding: 0;
}
.detail-table-notice__body {
  min-width: 0;
  margin: 0;
  padding: 0;
}
/* 详情表格：表头与表体单元格一律左对齐 */
.detail-panel__scroll :deep(.ant-table-thead > tr > th.ant-table-cell),
.detail-panel__scroll :deep(.ant-table-tbody > tr > td.ant-table-cell) {
  text-align: left !important;
}
/* 详情双图：略收标题与卡片底留白，配合 ECharts grid.bottom 收紧绘图区下方空白 */
.detail-panel__scroll .charts-grid .chart-title {
  margin-bottom: 8px;
}
.detail-panel__scroll .charts-grid .chart-card {
  padding: 18px 22px 12px;
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
