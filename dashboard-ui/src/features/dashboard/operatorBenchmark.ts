/**
 * 算子维度 CSV 业务规则（与规格说明一致）。
 * 由 `scripts/import-benchmark-data.ts` 与界面层共用。
 */

import { formatDisplayDateYmd } from '../../utils/formatDisplayDate'

export type OperatorTableRow = {
  shape: string
  dtype: string
  ic: number
  pt: number
  /** 原始备注，表格展示 */
  remarks: string
  /** 该行是否参与得分 / 计入口径 n / 代表延迟选取 */
  scoreEligible: boolean
  /** CSV date，详情横条可选展示 */
  date?: string
}

/** remarks 含 failed 或 Device Type Not Supported 时，ic 延迟不参与得分（仍展示行） */
export function remarksExcludeIcFromScore(remarks: string): boolean {
  const s = String(remarks ?? '').toLowerCase()
  return s.includes('failed') || s.includes('device type not supported')
}

/** remarks 非空且含 failed（大小写不敏感）— Shape 列前黄色警告标记 */
export function opRemarksContainsFailed(remarks: string | undefined): boolean {
  const s = String(remarks ?? '').trim()
  if (!s) return false
  return /failed/i.test(s)
}

export function canComputeOpRowScore(ic: number, pt: number, remarks: string): boolean {
  if (remarksExcludeIcFromScore(remarks)) return false
  if (!Number.isFinite(ic) || !Number.isFinite(pt) || ic <= 0 || pt <= 0) return false
  return true
}

/**
 * 行级得分（CSV：pt_latency_ms ÷ ic_latency_ms × 100）
 * 不符合条件时返回 null
 */
export function computeOpRowScore(ic: number, pt: number, remarks: string): number | null {
  if (!canComputeOpRowScore(ic, pt, remarks)) return null
  return (pt / ic) * 100
}

/** 延迟展示：小数点后 3 位 + ms（详情 KPI；概览卡 ic/pt 亦同口径） */
export function formatOpLatencyMs(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return '—'
  return `${ms.toFixed(3)}ms`
}

/** 概览卡左列 InfiniCore：最高分行的 ic_latency_ms（0.0XX ms 口径 = 三位小数） */
export function formatOpOverviewIcLatency(ms: number): string {
  return formatOpLatencyMs(ms)
}

/** 概览卡右列 PyTorch：最高分行的 pt_latency_ms（0.XXX ms 口径 = 三位小数） */
export function formatOpOverviewPtLatency(ms: number): string {
  return formatOpLatencyMs(ms)
}

export type OpPlatformOverview = {
  ownScore: number
  ownVal: string
  openVal: string
  n: number
  /** 概览卡「配置」：最优得分行的 shape · dtype */
  extra: string
  /** 详情「测试记录」副标题：全表 op_name 去重数，如「7 算子」 */
  opRecordSub: string
  adv: boolean
  advTxt: string
  /** 该平台算子 CSV 中的最大 date 字段（YYYY/MM/DD） */
  dataDate?: string
}

/**
 * 从扁平行列表生成算子维度概览卡指标：
 * - 左/右列延迟：得分最高行的 ic / pt，格式见 formatOpOverviewIcLatency / formatOpOverviewPtLatency
 * - 测试条数：CSV 行数减去 remarks 含 failed 的行
 * - 配置（extra）：得分最高行的 shape · dtype（如 M=3,N=3 · FP16）
 * - opRecordSub：全表 op 去重数「N 算子」（供详情「测试记录」副标题）
 */
export function buildOpPlatformOverview(
  flatRows: Array<{
    opKey: string
    shape: string
    dtype: string
    ic: number
    pt: number
    remarks: string
    date?: string
  }>,
): OpPlatformOverview | null {
  if (!flatRows.length) return null

  let best: (typeof flatRows)[0] | null = null
  let bestScore = -1
  for (const row of flatRows) {
    const s = computeOpRowScore(row.ic, row.pt, row.remarks)
    if (s != null && s > bestScore) {
      bestScore = s
      best = row
    }
  }

  const n = flatRows.filter((r) => !opRemarksContainsFailed(r.remarks)).length
  const opKeys = new Set(flatRows.map((r) => r.opKey))
  const opRecordSub = `${opKeys.size} 算子`
  const shapeDisp = String(best?.shape ?? '')
    .trim()
    .replace(/,\s*/g, ',')
  const dtypeDisp = String(best?.dtype ?? '')
    .trim()
    .toUpperCase()
  const extra =
    best != null ? [shapeDisp, dtypeDisp].filter(Boolean).join(' · ') || '—' : '—'

  const dates = flatRows
    .map((r) => r.date)
    .filter(Boolean)
    .sort() as string[]
  const dataDate = dates.length ? dates[dates.length - 1] : undefined

  const ownScore = best != null && bestScore >= 0 ? Math.round(bestScore) : 0
  const ownVal = best != null ? formatOpOverviewIcLatency(best.ic) : '—'
  const openVal = best != null ? formatOpOverviewPtLatency(best.pt) : '—'
  const adv = ownScore >= 100
  /** ownScore < 100 时不显示 advTxt：产品要求概览算子卡禁止再展示「部分配置待优化」之类弱化文案 */
  const advTxt =
    ownScore > 100
      ? `自研快 ${ownScore - 100}%`
      : ownScore >= 100
        ? '自研优于 PyTorch 基线'
        : ''

  return {
    ownScore,
    ownVal,
    openVal,
    n,
    extra,
    opRecordSub,
    adv,
    advTxt,
    dataDate,
  }
}

/** 某平台算子表（所有 op）中行内 `date` 字典序最大者（CSV date 字段） */
export function maxOperatorCsvDateForPlatform(
  platOps: Record<string, OperatorTableRow[]> | undefined,
): string | undefined {
  if (!platOps) return undefined
  let max = ''
  for (const rows of Object.values(platOps)) {
    for (const row of rows) {
      const d = String(row.date ?? '').trim()
      if (d && d > max) max = d
    }
  }
  return max || undefined
}

/** 侧栏等展示：统一为 YYYY-MM-DD */
export function formatOperatorCsvDateDisplay(d: string | undefined): string {
  return formatDisplayDateYmd(d)
}
