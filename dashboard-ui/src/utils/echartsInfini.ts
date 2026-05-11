/** ECharts option 构建 — 数值来自 @/data，与 dashboard_preview.html Chart.js 一致 */

import { CI_CHART_LABELS } from '@/data'

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
        symbolSize: 6,
        lineStyle: { color: '#00c853' },
      },
    ],
  }
}

type OpRow = { shape: string; dtype: string; ic: number; pt: number; scoreEligible?: boolean }

export function buildOpLineOption(rows: OpRow[], platColor: string) {
  const icSeries = rows.map((r) => (r.scoreEligible === false ? null : r.ic))
  const ptSeries = rows.map((r) => (r.scoreEligible === false ? null : r.pt))
  return {
    tooltip: { trigger: 'axis' as const },
    legend: { top: 0 },
    grid: { left: 48, right: 24, top: 40, bottom: 60 },
    xAxis: { type: 'category' as const, data: rows.map((r) => r.shape), boundaryGap: false },
    yAxis: { type: 'value' as const, name: '延迟 (ms)' },
    series: [
      {
        type: 'line' as const,
        name: 'InfiniCore',
        data: icSeries,
        smooth: true,
        symbolSize: 8,
        connectNulls: false,
        lineStyle: { color: platColor },
      },
      {
        type: 'line' as const,
        name: 'PyTorch',
        data: ptSeries,
        smooth: true,
        symbolSize: 8,
        connectNulls: false,
        lineStyle: { color: '#999' },
      },
    ],
  }
}

export function buildOpBarAvgOption(opKeys: string[], scores: number[], platColor: string) {
  return {
    tooltip: { trigger: 'axis' as const },
    grid: { left: 48, right: 24, top: 28, bottom: 48 },
    xAxis: { type: 'category' as const, data: opKeys },
    yAxis: { type: 'value' as const, name: '得分', min: 0 },
    series: [
      {
        type: 'bar' as const,
        name: '平均得分',
        data: scores,
        itemStyle: { color: platColor + 'cc', borderRadius: [6, 6, 0, 0] },
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
  return {
    tooltip: { trigger: 'axis' as const },
    legend: { top: 0 },
    grid: { left: 56, right: 24, top: 40, bottom: 48 },
    xAxis: {
      type: 'category' as const,
      data: prefillRows.map((r) => `bs${r.batch} in${r.inLen}`),
    },
    yAxis: { type: 'value' as const, name: 'tokens/s' },
    series: [
      {
        type: 'bar' as const,
        name: 'Prefill TPS',
        data: prefillRows.map((r) => r.tps),
        itemStyle: { color: platColor + 'cc', borderRadius: 5 },
      },
      {
        type: 'bar' as const,
        name: 'A100 Prefill',
        data: nvPrefill.map((r) => r.tps),
        itemStyle: { color: '#76b90044', borderRadius: 5 },
      },
    ],
  }
}

export function buildInferDecodeBarOption(
  decodeRows: InferRow[],
  nvDecode: InferRow[],
  platColor: string,
) {
  return {
    tooltip: { trigger: 'axis' as const },
    legend: { top: 0 },
    grid: { left: 56, right: 24, top: 40, bottom: 48 },
    xAxis: {
      type: 'category' as const,
      data: decodeRows.map((r) => `bs${r.batch} in${r.inLen}`),
    },
    yAxis: { type: 'value' as const, name: 'tokens/s' },
    series: [
      {
        type: 'bar' as const,
        name: 'Decode TPS',
        data: decodeRows.map((r) => r.tps),
        itemStyle: { color: platColor + '99', borderRadius: 5 },
      },
      {
        type: 'bar' as const,
        name: 'A100 Decode',
        data: nvDecode.map((r) => r.tps),
        itemStyle: { color: '#76b90044', borderRadius: 5 },
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
  return {
    tooltip: { trigger: 'axis' as const },
    legend: { top: 0 },
    grid: { left: 56, right: 24, top: 40, bottom: 48 },
    xAxis: { type: 'category' as const, data: categories },
    yAxis: { type: 'value' as const, name: 'tokens/s' },
    series: [
      {
        type: 'bar' as const,
        name: 'Prefill TPS',
        data: platVals,
        itemStyle: { color: platColor + 'cc', borderRadius: 5 },
      },
      {
        type: 'bar' as const,
        name: 'NVIDIA Prefill',
        data: nvVals,
        itemStyle: { color: '#76b90044', borderRadius: 5 },
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
  return {
    tooltip: { trigger: 'axis' as const },
    legend: { top: 0 },
    grid: { left: 56, right: 24, top: 40, bottom: 48 },
    xAxis: { type: 'category' as const, data: categories },
    yAxis: { type: 'value' as const, name: 'tokens/s' },
    series: [
      {
        type: 'bar' as const,
        name: 'Decode TPS',
        data: platVals,
        itemStyle: { color: platColor + '99', borderRadius: 5 },
      },
      {
        type: 'bar' as const,
        name: 'NVIDIA Decode',
        data: nvVals,
        itemStyle: { color: '#76b90044', borderRadius: 5 },
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
  return {
    tooltip: { trigger: 'axis' as const },
    legend: { top: 0 },
    grid: { left: 64, right: 24, top: 40, bottom: 72 },
    xAxis: {
      type: 'category' as const,
      data: rows.map((r) => `${r.framework}·${r.model}`),
      axisLabel: { rotate: 25, fontSize: 10 },
    },
    yAxis: { type: 'value' as const, name: 'tokens/process/s' },
    series: [
      {
        type: 'bar' as const,
        name: '实测吞吐 tpps',
        data: rows.map((r) => r.tps),
        itemStyle: { color: platColor + 'cc', borderRadius: 6 },
      },
      {
        type: 'bar' as const,
        name: 'NVIDIA 基线',
        data: rows.map((r) => r.baseline),
        itemStyle: { color: '#76b90055', borderRadius: 6 },
      },
    ],
  }
}

export function buildTrainBarVs(rows: TrainRow[]) {
  return {
    tooltip: { trigger: 'axis' as const },
    grid: { left: 56, right: 24, top: 28, bottom: 72 },
    xAxis: {
      type: 'category' as const,
      data: rows.map((r) => `${r.framework}·${r.model}`),
      axisLabel: { rotate: 25, fontSize: 10 },
    },
    yAxis: { type: 'value' as const, name: '% vs NVIDIA', min: 0 },
    series: [
      {
        type: 'bar' as const,
        name: 'vs NVIDIA (%)',
        data: rows.map((r) => ({
          value: r.vsA100,
          itemStyle: {
            color: r.vsA100 >= 100 ? '#2e7d32cc' : '#e6510099',
            borderRadius: 6,
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
  return {
    tooltip: { trigger: 'axis' as const },
    legend: { top: 0 },
    grid: { left: 52, right: 24, top: 40, bottom: 48 },
    xAxis: {
      type: 'category' as const,
      data: rows.map((r) => r.commType + ' ' + r.nGpu + 'GPU'),
    },
    yAxis: { type: 'value' as const, name: 'GB/s' },
    series: [
      {
        type: 'bar' as const,
        name: '自研带宽 GB/s',
        data: rows.map((r) => r.bw),
        itemStyle: { color: platColor + 'cc', borderRadius: 6 },
      },
      {
        type: 'bar' as const,
        name: 'NVIDIA 基线',
        data: rows.map((r) => r.baseline),
        itemStyle: { color: '#76b90055', borderRadius: 6 },
      },
    ],
  }
}

export function buildCommBarVs(rows: CommRow[]) {
  return {
    tooltip: { trigger: 'axis' as const },
    grid: { left: 52, right: 24, top: 28, bottom: 48 },
    xAxis: { type: 'category' as const, data: rows.map((r) => r.commType) },
    yAxis: { type: 'value' as const, name: '% vs NVIDIA', min: 0 },
    series: [
      {
        type: 'bar' as const,
        name: 'vs NVIDIA (%)',
        data: rows.map((r) => ({
          value: r.vsA100,
          itemStyle: {
            color: r.vsA100 >= 100 ? '#2e7d32cc' : '#e6510099',
            borderRadius: 6,
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
  return {
    tooltip: { trigger: 'axis' as const },
    legend: { top: 0 },
    grid: { left: 52, right: 24, top: 40, bottom: 48 },
    xAxis: { type: 'category' as const, data: valid.map((r) => r.model) },
    yAxis: { type: 'value' as const, name: 'GB/s' },
    series: [
      {
        type: 'bar' as const,
        name: '均值 GB/s',
        data: valid.map((r) => r.avg as number),
        itemStyle: { color: platColor + 'cc', borderRadius: 6 },
      },
      {
        type: 'bar' as const,
        name: 'NVIDIA A100 基线',
        data: valid.map(() => nvidiaAvg),
        itemStyle: { color: '#76b90055', borderRadius: 6 },
      },
    ],
  }
}

export function buildBwBarModes(bestRow: BwRow, nvidiaRow: BwRow, platColor: string) {
  const modes = ['add', 'copy', 'scale', 'triad'] as const
  return {
    tooltip: { trigger: 'axis' as const },
    legend: { top: 0 },
    grid: { left: 52, right: 24, top: 40, bottom: 40 },
    xAxis: { type: 'category' as const, data: [...modes] },
    yAxis: { type: 'value' as const, name: 'GB/s' },
    series: [
      {
        type: 'bar' as const,
        name: bestRow.model || '自研',
        data: modes.map((m) => bestRow[m] as number),
        itemStyle: { color: platColor + 'cc', borderRadius: 6 },
      },
      {
        type: 'bar' as const,
        name: 'NVIDIA A100（四模式参考）',
        data: modes.map((m) => nvidiaRow[m] as number),
        itemStyle: { color: '#76b90055', borderRadius: 6 },
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
  return {
    tooltip: { trigger: 'axis' as const },
    legend: { top: 0, textStyle: { fontSize: 11 } },
    grid: { left: 56, right: 24, top: 36, bottom: 48 },
    xAxis: { type: 'category' as const, data: plats.map((p) => p.name) },
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
        data: cards.map((c, i) => ({
          value: c.ownScore != null ? Number((c.ownScore / 100).toFixed(2)) : 0,
          itemStyle: { color: plats[i].color + 'cc', borderRadius: 6 },
        })),
      },
      {
        type: 'bar' as const,
        name: '开源基准（×1.0）',
        data: cards.map(() => ({
          value: 1,
          itemStyle: { color: '#bbbbbb55', borderRadius: 6 },
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
  return {
    tooltip: { trigger: 'axis' as const },
    grid: { left: 52, right: 24, top: 28, bottom: 48 },
    xAxis: { type: 'category' as const, data: names },
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
        data: latencies.map((v, i) => ({
          value: v,
          itemStyle: { color: colors[i] + 'cc', borderRadius: 6 },
        })),
      },
    ],
  }
}
