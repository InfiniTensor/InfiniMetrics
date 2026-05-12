/** ECharts option 构建 — 数值来自 @/data，与 dashboard_preview.html Chart.js 一致 */

import { CI_CHART_LABELS } from '@/data'

/** 详情区算子延迟/得分柱/CI 折线统一主蓝（与各算子平均得分柱同色） */
const CHART_PRIMARY_BLUE = '#3182ce'

/** 详情页柱状图单柱最大宽度（px） */
const DETAIL_BAR_MAX_WIDTH = 20

function maxCategoryLabelLen(categories: string[]): number {
  return categories.reduce((m, s) => Math.max(m, String(s).length), 0)
}

/** 类目轴能横排展示时不旋转（如仅 1 条 megatron-…）；否则再按密度斜排 */
function categoryBarLabelsFitHorizontal(categories: string[]): boolean {
  const n = categories.length
  if (n <= 0) return true
  if (n === 1) return true
  const maxLen = maxCategoryLabelLen(categories)
  if (n <= 2 && maxLen <= 28) return true
  if (n <= 4 && maxLen <= 14) return true
  if (n <= 6 && maxLen <= 10) return true
  if (n <= 8 && maxLen <= 8) return true
  return false
}

/** 详情柱图类目轴：能放下则横排，否则随条数加大旋转与底部留白 */
function categoryBarXAxisUi(categories: string[]): {
  gridBottom: number
  axisLabel: { interval: 0; rotate: number; fontSize: number; margin: number }
} {
  const n = categories.length
  if (categoryBarLabelsFitHorizontal(categories)) {
    return {
      gridBottom: 46,
      axisLabel: { interval: 0, rotate: 0, fontSize: 11, margin: 8 },
    }
  }
  if (n <= 8) {
    return {
      gridBottom: 62,
      axisLabel: { interval: 0, rotate: 28, fontSize: 10, margin: 8 },
    }
  }
  if (n <= 15) {
    return {
      gridBottom: 82,
      axisLabel: { interval: 0, rotate: 42, fontSize: 9, margin: 10 },
    }
  }
  return {
    gridBottom: 112,
    axisLabel: { interval: 0, rotate: 55, fontSize: 8, margin: 10 },
  }
}

export function buildCiLineOption(seriesData: number[]) {
  return {
    tooltip: { trigger: 'axis' as const },
    grid: { left: 48, right: 24, top: 28, bottom: 32 },
    xAxis: { type: 'category' as const, data: [...CI_CHART_LABELS], boundaryGap: false },
    yAxis: { type: 'value' as const, min: 0 },
    series: [
      {
        type: 'line' as const,
        name: '运行次数',
        data: seriesData,
        smooth: true,
        symbolSize: 3,
        lineStyle: { color: CHART_PRIMARY_BLUE, width: 1 },
      },
    ],
  }
}

type OpRow = { shape: string; dtype: string; ic: number; pt: number; scoreEligible?: boolean }

export function buildOpLineOption(rows: OpRow[]) {
  const icSeries = rows.map((r) => (r.scoreEligible === false ? null : r.ic))
  const ptSeries = rows.map((r) => (r.scoreEligible === false ? null : r.pt))
  return {
    tooltip: { trigger: 'axis' as const },
    legend: { top: 2, right: 8 },
    grid: { left: 48, right: 24, top: 54, bottom: 36 },
    xAxis: { type: 'category' as const, data: rows.map((r) => r.shape), boundaryGap: false },
    yAxis: { type: 'value' as const, name: '延迟 (ms)' },
    series: [
      {
        type: 'line' as const,
        name: 'InfiniCore',
        data: icSeries,
        smooth: true,
        symbolSize: 3,
        connectNulls: false,
        lineStyle: { color: CHART_PRIMARY_BLUE, width: 1 },
      },
      {
        type: 'line' as const,
        name: 'PyTorch',
        data: ptSeries,
        smooth: true,
        symbolSize: 3,
        connectNulls: false,
        lineStyle: { color: '#999', width: 1 },
      },
    ],
  }
}

export function buildOpBarAvgOption(opKeys: string[], scores: number[]) {
  const xUi = categoryBarXAxisUi(opKeys)
  return {
    tooltip: { trigger: 'axis' as const },
    grid: { left: 48, right: 24, top: 28, bottom: xUi.gridBottom },
    xAxis: {
      type: 'category' as const,
      data: opKeys,
      axisLabel: xUi.axisLabel,
    },
    yAxis: { type: 'value' as const, name: '得分', min: 0 },
    series: [
      {
        type: 'bar' as const,
        name: '平均得分',
        data: scores,
        barMaxWidth: DETAIL_BAR_MAX_WIDTH,
        barCategoryGap: '42%',
        itemStyle: { color: CHART_PRIMARY_BLUE + 'cc', borderRadius: [2, 2, 0, 0] },
      },
    ],
  }
}

type InferRow = { batch: number; inLen: number; tps: number; ttft?: number }

export function buildInferPrefillBarOption(
  prefillRows: InferRow[],
  nvPrefill: InferRow[],
  platColor: string,
) {
  const cats = prefillRows.map((r) => `bs${r.batch} in${r.inLen}`)
  const xUi = categoryBarXAxisUi(cats)
  return {
    tooltip: { trigger: 'axis' as const },
    legend: { top: 2, right: 8 },
    grid: { left: 56, right: 24, top: 56, bottom: xUi.gridBottom },
    xAxis: {
      type: 'category' as const,
      data: cats,
      axisLabel: xUi.axisLabel,
    },
    yAxis: { type: 'value' as const, name: 'tokens/s' },
    series: [
      {
        type: 'bar' as const,
        name: 'Prefill TPS',
        data: prefillRows.map((r) => r.tps),
        barMaxWidth: DETAIL_BAR_MAX_WIDTH,
        itemStyle: { color: platColor + 'cc', borderRadius: [2, 2, 0, 0] },
      },
      {
        type: 'bar' as const,
        name: 'A100 Prefill',
        data: nvPrefill.map((r) => r.tps),
        barMaxWidth: DETAIL_BAR_MAX_WIDTH,
        itemStyle: { color: '#76b90044', borderRadius: [2, 2, 0, 0] },
      },
    ],
  }
}

export function buildInferDecodeBarOption(
  decodeRows: InferRow[],
  nvDecode: InferRow[],
  platColor: string,
) {
  const cats = decodeRows.map((r) => `bs${r.batch} in${r.inLen}`)
  const xUi = categoryBarXAxisUi(cats)
  return {
    tooltip: { trigger: 'axis' as const },
    legend: { top: 2, right: 8 },
    grid: { left: 56, right: 24, top: 56, bottom: xUi.gridBottom },
    xAxis: {
      type: 'category' as const,
      data: cats,
      axisLabel: xUi.axisLabel,
    },
    yAxis: { type: 'value' as const, name: 'tokens/s' },
    series: [
      {
        type: 'bar' as const,
        name: 'Decode TPS',
        data: decodeRows.map((r) => r.tps),
        barMaxWidth: DETAIL_BAR_MAX_WIDTH,
        itemStyle: { color: platColor + '99', borderRadius: [2, 2, 0, 0] },
      },
      {
        type: 'bar' as const,
        name: 'A100 Decode',
        data: nvDecode.map((r) => r.tps),
        barMaxWidth: DETAIL_BAR_MAX_WIDTH,
        itemStyle: { color: '#76b90044', borderRadius: [2, 2, 0, 0] },
      },
    ],
  }
}

export function buildInferPrefillBarAligned(
  categories: string[],
  platVals: number[],
  nvVals: number[],
  platColor: string,
) {
  const xUi = categoryBarXAxisUi(categories)
  return {
    tooltip: { trigger: 'axis' as const },
    legend: { top: 2, right: 8 },
    grid: { left: 56, right: 24, top: 56, bottom: xUi.gridBottom },
    xAxis: { type: 'category' as const, data: categories, axisLabel: xUi.axisLabel },
    yAxis: { type: 'value' as const, name: 'tokens/s' },
    series: [
      {
        type: 'bar' as const,
        name: 'Prefill TPS',
        data: platVals,
        barMaxWidth: DETAIL_BAR_MAX_WIDTH,
        itemStyle: { color: platColor + 'cc', borderRadius: [2, 2, 0, 0] },
      },
      {
        type: 'bar' as const,
        name: 'NVIDIA Prefill',
        data: nvVals,
        barMaxWidth: DETAIL_BAR_MAX_WIDTH,
        itemStyle: { color: '#76b90044', borderRadius: [2, 2, 0, 0] },
      },
    ],
  }
}

export function buildInferDecodeBarAligned(
  categories: string[],
  platVals: number[],
  nvVals: number[],
  platColor: string,
) {
  const xUi = categoryBarXAxisUi(categories)
  return {
    tooltip: { trigger: 'axis' as const },
    legend: { top: 2, right: 8 },
    grid: { left: 56, right: 24, top: 56, bottom: xUi.gridBottom },
    xAxis: { type: 'category' as const, data: categories, axisLabel: xUi.axisLabel },
    yAxis: { type: 'value' as const, name: 'tokens/s' },
    series: [
      {
        type: 'bar' as const,
        name: 'Decode TPS',
        data: platVals,
        barMaxWidth: DETAIL_BAR_MAX_WIDTH,
        itemStyle: { color: platColor + '99', borderRadius: [2, 2, 0, 0] },
      },
      {
        type: 'bar' as const,
        name: 'NVIDIA Decode',
        data: nvVals,
        barMaxWidth: DETAIL_BAR_MAX_WIDTH,
        itemStyle: { color: '#76b90044', borderRadius: [2, 2, 0, 0] },
      },
    ],
  }
}

type TrainRow = {
  framework: string
  model: string
  parallel: string
  dtype: string
  flashAttn: string
  tps: number
  baseline: number
  vsA100: number
  note: string
}

export function buildTrainBarThroughput(rows: TrainRow[], platColor: string) {
  const categories = rows.map((r) => `${r.framework}·${r.model}`)
  const xUi = categoryBarXAxisUi(categories)
  return {
    tooltip: { trigger: 'axis' as const },
    legend: { top: 2, right: 8 },
    grid: { left: 64, right: 24, top: 56, bottom: xUi.gridBottom },
    xAxis: {
      type: 'category' as const,
      data: categories,
      axisLabel: xUi.axisLabel,
    },
    yAxis: { type: 'value' as const, name: 'tokens/process/s' },
    series: [
      {
        type: 'bar' as const,
        name: '实测吞吐 tpps',
        data: rows.map((r) => r.tps),
        barMaxWidth: DETAIL_BAR_MAX_WIDTH,
        itemStyle: { color: platColor + 'cc', borderRadius: [2, 2, 0, 0] },
      },
      {
        type: 'bar' as const,
        name: 'NVIDIA 基线',
        data: rows.map((r) => r.baseline),
        barMaxWidth: DETAIL_BAR_MAX_WIDTH,
        itemStyle: { color: '#76b90055', borderRadius: [2, 2, 0, 0] },
      },
    ],
  }
}

export function buildTrainBarVs(rows: TrainRow[]) {
  const categories = rows.map((r) => `${r.framework}·${r.model}`)
  const xUi = categoryBarXAxisUi(categories)
  return {
    tooltip: { trigger: 'axis' as const },
    grid: { left: 56, right: 24, top: 28, bottom: xUi.gridBottom },
    xAxis: {
      type: 'category' as const,
      data: categories,
      axisLabel: xUi.axisLabel,
    },
    yAxis: { type: 'value' as const, name: '% vs NVIDIA', min: 0 },
    series: [
      {
        type: 'bar' as const,
        name: 'vs NVIDIA (%)',
        barMaxWidth: DETAIL_BAR_MAX_WIDTH,
        data: rows.map((r) => ({
          value: r.vsA100,
          itemStyle: {
            color: r.vsA100 >= 100 ? '#2e7d32cc' : '#e6510099',
            borderRadius: [2, 2, 0, 0],
          },
        })),
      },
    ],
  }
}

type CommRow = {
  linkType: string
  commType: string
  nGpu: number
  bw: number
  baseline: number
  vsA100: number
  note: string
}

export function buildCommBarBw(rows: CommRow[], platColor: string) {
  const categories = rows.map((r) => r.commType + ' ' + r.nGpu + 'GPU')
  const xUi = categoryBarXAxisUi(categories)
  return {
    tooltip: { trigger: 'axis' as const },
    legend: { top: 2, right: 8 },
    grid: { left: 52, right: 24, top: 56, bottom: xUi.gridBottom },
    xAxis: {
      type: 'category' as const,
      data: categories,
      axisLabel: xUi.axisLabel,
    },
    yAxis: { type: 'value' as const, name: 'GB/s' },
    series: [
      {
        type: 'bar' as const,
        name: '自研带宽 GB/s',
        data: rows.map((r) => r.bw),
        barMaxWidth: DETAIL_BAR_MAX_WIDTH,
        itemStyle: { color: platColor + 'cc', borderRadius: [2, 2, 0, 0] },
      },
      {
        type: 'bar' as const,
        name: 'NVIDIA 基线',
        data: rows.map((r) => r.baseline),
        barMaxWidth: DETAIL_BAR_MAX_WIDTH,
        itemStyle: { color: '#76b90055', borderRadius: [2, 2, 0, 0] },
      },
    ],
  }
}

export function buildCommBarVs(rows: CommRow[]) {
  const categories = rows.map((r) => r.commType)
  const xUi = categoryBarXAxisUi(categories)
  return {
    tooltip: { trigger: 'axis' as const },
    grid: { left: 52, right: 24, top: 28, bottom: xUi.gridBottom },
    xAxis: { type: 'category' as const, data: categories, axisLabel: xUi.axisLabel },
    yAxis: { type: 'value' as const, name: '% vs NVIDIA', min: 0 },
    series: [
      {
        type: 'bar' as const,
        name: 'vs NVIDIA (%)',
        barMaxWidth: DETAIL_BAR_MAX_WIDTH,
        data: rows.map((r) => ({
          value: r.vsA100,
          itemStyle: {
            color: r.vsA100 >= 100 ? '#2e7d32cc' : '#e6510099',
            borderRadius: [2, 2, 0, 0],
          },
        })),
      },
    ],
  }
}

type BwRow = {
  model: string
  add: number | null
  copy: number | null
  scale: number | null
  triad: number | null
  avg: number | null
}

export function buildBwBarAvg(
  rows: BwRow[],
  platColor: string,
  nvidiaAvg: number,
) {
  const valid = rows.filter((r) => r.avg != null)
  const categories = valid.map((r) => r.model)
  const xUi = categoryBarXAxisUi(categories)
  return {
    tooltip: { trigger: 'axis' as const },
    legend: { top: 2, right: 8 },
    grid: { left: 52, right: 24, top: 56, bottom: xUi.gridBottom },
    xAxis: { type: 'category' as const, data: categories, axisLabel: xUi.axisLabel },
    yAxis: { type: 'value' as const, name: 'GB/s' },
    series: [
      {
        type: 'bar' as const,
        name: '均值 GB/s',
        data: valid.map((r) => r.avg as number),
        barMaxWidth: DETAIL_BAR_MAX_WIDTH,
        itemStyle: { color: platColor + 'cc', borderRadius: [2, 2, 0, 0] },
      },
      {
        type: 'bar' as const,
        name: 'NVIDIA A100 基线',
        data: valid.map(() => nvidiaAvg),
        barMaxWidth: DETAIL_BAR_MAX_WIDTH,
        itemStyle: { color: '#76b90055', borderRadius: [2, 2, 0, 0] },
      },
    ],
  }
}

export function buildBwBarModes(bestRow: BwRow, nvidiaRow: BwRow, platColor: string) {
  const modes = ['add', 'copy', 'scale', 'triad'] as const
  const categories = ['add', 'copy', 'scale', 'triad']
  const xUi = categoryBarXAxisUi(categories)
  return {
    tooltip: { trigger: 'axis' as const },
    legend: { top: 2, right: 8 },
    grid: { left: 52, right: 24, top: 56, bottom: xUi.gridBottom },
    xAxis: { type: 'category' as const, data: categories, axisLabel: xUi.axisLabel },
    yAxis: { type: 'value' as const, name: 'GB/s' },
    series: [
      {
        type: 'bar' as const,
        name: bestRow.model || '自研',
        data: modes.map((m) => bestRow[m] as number),
        barMaxWidth: DETAIL_BAR_MAX_WIDTH,
        itemStyle: { color: platColor + 'cc', borderRadius: [2, 2, 0, 0] },
      },
      {
        type: 'bar' as const,
        name: 'NVIDIA A100（四模式参考）',
        data: modes.map((m) => nvidiaRow[m] as number),
        barMaxWidth: DETAIL_BAR_MAX_WIDTH,
        itemStyle: { color: '#76b90055', borderRadius: [2, 2, 0, 0] },
      },
    ],
  }
}

type CardLite = {
  key: string
  ownScore: number | null
  ownVal?: string | null
}

type PlatLite = { key: string; name: string; color: string }

export function buildCompareScoreBar(cards: CardLite[], plats: PlatLite[]) {
  const categories = plats.map((p) => p.name)
  const xUi = categoryBarXAxisUi(categories)
  return {
    tooltip: { trigger: 'axis' as const },
    legend: { top: 2, right: 8, textStyle: { fontSize: 11 } },
    grid: { left: 56, right: 24, top: 50, bottom: xUi.gridBottom },
    xAxis: { type: 'category' as const, data: categories, axisLabel: xUi.axisLabel },
    yAxis: {
      type: 'value' as const,
      name: '提速倍率（× 相对开源基准）',
      min: 0,
      axisLabel: { formatter: (v: number) => v + '×' },
      splitLine: { lineStyle: { color: '#f0f0f0' } },
    },
    series: [
      {
        type: 'bar' as const,
        name: '自研提速倍率（×）',
        barMaxWidth: DETAIL_BAR_MAX_WIDTH,
        data: cards.map((c, i) => ({
          value: c.ownScore != null ? Number((c.ownScore / 100).toFixed(2)) : 0,
          itemStyle: { color: plats[i].color + 'cc', borderRadius: [2, 2, 0, 0] },
        })),
      },
      {
        type: 'bar' as const,
        name: '开源基准（×1.0）',
        barMaxWidth: DETAIL_BAR_MAX_WIDTH,
        data: cards.map(() => ({
          value: 1,
          itemStyle: { color: '#bbbbbb55', borderRadius: [2, 2, 0, 0] },
        })),
      },
    ],
  }
}

export function buildCompareLatencyBar(
  names: string[],
  latencies: number[],
  colors: string[],
) {
  const xUi = categoryBarXAxisUi(names)
  return {
    tooltip: { trigger: 'axis' as const },
    grid: { left: 52, right: 24, top: 28, bottom: xUi.gridBottom },
    xAxis: { type: 'category' as const, data: names, axisLabel: xUi.axisLabel },
    yAxis: {
      type: 'value' as const,
      name: 'ms，越低越好',
      min: 0,
      splitLine: { lineStyle: { color: '#f0f0f0' } },
    },
    series: [
      {
        type: 'bar' as const,
        name: '自研延迟 ms',
        barMaxWidth: DETAIL_BAR_MAX_WIDTH,
        data: latencies.map((v, i) => ({
          value: v,
          itemStyle: { color: colors[i] + 'cc', borderRadius: [2, 2, 0, 0] },
        })),
      },
    ],
  }
}
