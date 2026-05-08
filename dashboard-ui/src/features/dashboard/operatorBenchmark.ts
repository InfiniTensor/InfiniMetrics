/**
 * 算子维度 CSV 业务规则（与规格说明一致）。
 * 由 `scripts/import-benchmark-data.ts` 与界面层共用。
 */

export type OperatorTableRow = {
  shape: string
  dtype: string
  ic: number
  pt: number
  /** 原始备注，表格展示 */
  remarks: string
  /** 该行是否参与得分 / 计入口径 n / 代表延迟选取 */
  scoreEligible: boolean
}

/** remarks 含 failed 或 Device Type Not Supported 时，ic 延迟不参与得分（仍展示行） */
export function remarksExcludeIcFromScore(remarks: string): boolean {
  const s = String(remarks ?? '').toLowerCase()
  return s.includes('failed') || s.includes('device type not supported')
}

export function canComputeOpRowScore(ic: number, pt: number, remarks: string): boolean {
  if (remarksExcludeIcFromScore(remarks)) return false
  if (!Number.isFinite(ic) || !Number.isFinite(pt) || ic <= 0 || pt <= 0) return false
  return true
}

/** 行级得分：pt / ic * 100；不符合条件时返回 null */
export function computeOpRowScore(ic: number, pt: number, remarks: string): number | null {
  if (!canComputeOpRowScore(ic, pt, remarks)) return null
  return (pt / ic) * 100
}

/** 概览卡延迟展示：0.0XX ms（三位小数） */
export function formatOpLatencyMs(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return '—'
  return `${ms.toFixed(3)}ms`
}

export type OpPlatformOverview = {
  ownScore: number
  ownVal: string
  openVal: string
  n: number
  extra: string
  adv: boolean
  advTxt: string
  /** 该平台算子 CSV 中的最大 date 字段（YYYY/MM/DD） */
  dataDate?: string
}

/**
 * 从扁平行列表生成概览卡指标（与规格：代表行为最高分那一行等一致）。
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

  const n = flatRows.filter((r) => canComputeOpRowScore(r.ic, r.pt, r.remarks)).length
  const opKeys = new Set(flatRows.map((r) => r.opKey))
  const extra = `${opKeys.size} 算子`

  const dates = flatRows
    .map((r) => r.date)
    .filter(Boolean)
    .sort() as string[]
  const dataDate = dates.length ? dates[dates.length - 1] : undefined

  const ownScore = best != null && bestScore >= 0 ? Math.round(bestScore) : 0
  const ownVal = best != null ? formatOpLatencyMs(best.ic) : '—'
  const openVal = best != null ? formatOpLatencyMs(best.pt) : '—'
  const adv = ownScore >= 100
  const advTxt =
    ownScore > 100
      ? `自研快 ${ownScore - 100}%`
      : ownScore >= 100
        ? '自研优于 PyTorch 基线'
        : '部分配置待优化'

  return {
    ownScore,
    ownVal,
    openVal,
    n,
    extra,
    adv,
    advTxt,
    dataDate,
  }
}
