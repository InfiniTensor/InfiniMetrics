/**
 * 推理维度 CSV 与展示指标（与 infer CSV 规格一致）。
 */

export const INFER_FRAMEWORK = 'InfiniLM' as const

export type InferPrefillRow = {
  configKey: string
  batch: number
  inLen: number
  outLen: number
  model: string
  dtype: string
  nGpu: number
  remarks: string
  tps: number
  ttft?: number
  /** 单步 decode 延迟参考（ms） */
  decodeLatencyMs?: number
  /** 相对 NVIDIA 同配置 Prefill TPS 的百分比，四舍五入 */
  vsNvidia: number | null
  /** 同配置 NVIDIA Prefill TPS（柱状对比条） */
  nvidiaBaselineTps: number | null
  framework: typeof INFER_FRAMEWORK
}

export type InferDecodeRow = {
  configKey: string
  batch: number
  inLen: number
  outLen: number
  model: string
  dtype: string
  nGpu: number
  remarks: string
  tps: number
  decodeLatencyMs?: number
  vsNvidia: number | null
  nvidiaBaselineTps: number | null
  framework: typeof INFER_FRAMEWORK
}

export function inferConfigKey(batch: number, inLen: number, outLen: number): string {
  return `${batch}:${inLen}:${outLen}`
}

export function normalizeInferModel(model: string): string {
  return String(model || '')
    .trim()
    .toUpperCase()
}

/** 吞吐展示：大数用 K（与既有卡片风格一致） */
export function formatInferTokPerS(tps: number): string {
  if (!Number.isFinite(tps) || tps < 0) return '—'
  if (tps >= 1000) return `${(tps / 1000).toFixed(1)}K tok/s`
  return `${Math.round(tps)} tok/s`
}

export function inferVsPercent(platTps: number, nvTps: number): number | null {
  if (!Number.isFinite(platTps) || !Number.isFinite(nvTps) || nvTps <= 0) return null
  return Math.round((platTps / nvTps) * 100)
}

export type InferFlatRow = {
  plat: string
  batch: number
  inLen: number
  outLen: number
  model: string
  dtype: string
  nGpu: number
  remarks: string
  date: string
  ttftMs: number | null
  decodeMs: number | null
  prefillTps: number | null
  decodeTps: number | null
}

export type InferCardMetrics = {
  ownScore: number
  openScore: number
  ownVal: string
  openVal: string
  n: number
  extra: string
  adv: boolean
  advTxt: string
  dataDate?: string
}

/**
 * 从扁平行计算概览卡（相对 NVIDIA 全表最大 Prefill / Decode）。
 */
export function buildInferCardMetrics(
  rows: InferFlatRow[],
  nvMaxPrefill: number,
  nvMaxDecode: number,
): InferCardMetrics | null {
  if (!rows.length) return null
  let maxP: number | null = null
  let maxD: number | null = null
  let argP: InferFlatRow | null = null
  let minTtft = Infinity
  let argTtftRow: InferFlatRow | null = null
  const dates: string[] = []

  for (const r of rows) {
    if (r.date) dates.push(r.date)
    if (r.prefillTps != null && Number.isFinite(r.prefillTps)) {
      if (maxP == null || r.prefillTps > maxP) {
        maxP = r.prefillTps
        argP = r
      }
    }
    if (r.decodeTps != null && Number.isFinite(r.decodeTps)) {
      if (maxD == null || r.decodeTps > maxD) maxD = r.decodeTps
    }
    if (r.ttftMs != null && r.ttftMs > 0 && r.ttftMs < minTtft) {
      minTtft = r.ttftMs
      argTtftRow = r
    }
  }

  const n = rows.filter((r) => r.prefillTps != null || r.decodeTps != null).length
  const dataDate = dates.length ? [...dates].sort().at(-1) : undefined

  const nvP = nvMaxPrefill > 0 ? nvMaxPrefill : 1
  const nvD = nvMaxDecode > 0 ? nvMaxDecode : 1
  const ownScore = maxP != null ? Math.round((maxP / nvP) * 100) : 0
  const openScore = maxD != null ? Math.round((maxD / nvD) * 100) : 0
  const ownVal = maxP != null ? formatInferTokPerS(maxP) : '—'
  const openVal = maxD != null ? formatInferTokPerS(maxD) : '—'
  const extra =
    argP != null
      ? `batch=${argP.batch} in=${argP.inLen} out=${argP.outLen}`
      : argTtftRow != null
        ? `batch=${argTtftRow.batch} in=${argTtftRow.inLen}`
        : ''

  const adv = ownScore >= 100
  const advTxt =
    ownScore >= 100
      ? `Prefill 相对 NVIDIA ${ownScore}%`
      : ownScore >= 70
        ? `Prefill 相对 NVIDIA ${ownScore}%（可优化）`
        : `Prefill 相对 NVIDIA ${ownScore}%`

  return { ownScore, openScore, ownVal, openVal, n, extra, adv, advTxt, dataDate }
}

/** 供图表：按 configKey 排序后对齐两类行（缺失填 0） */
export function alignInferSeries<
  T extends { configKey: string; batch: number; inLen: number; outLen: number; tps: number },
>(plat: T[], nv: T[]): { categories: string[]; platVals: number[]; nvVals: number[] } {
  const keys = new Set<string>()
  for (const r of plat) keys.add(r.configKey)
  for (const r of nv) keys.add(r.configKey)
  const sorted = [...keys].sort((a, b) => {
    const [ab, ai, ao] = a.split(':').map(Number)
    const [bb, bi, bo] = b.split(':').map(Number)
    if (ab !== bb) return ab - bb
    if (ai !== bi) return ai - bi
    return ao - bo
  })
  const pm = new Map(plat.map((r) => [r.configKey, r.tps]))
  const nm = new Map(nv.map((r) => [r.configKey, r.tps]))
  return {
    categories: sorted.map((k) => {
      const [b, i] = k.split(':')
      return `bs${b} in${i}`
    }),
    platVals: sorted.map((k) => pm.get(k) ?? 0),
    nvVals: sorted.map((k) => nm.get(k) ?? 0),
  }
}

export function filterInferRows<
  T extends { batch: number; inLen: number },
>(rows: T[], batchPill: string | undefined, inLenPill: string | undefined): T[] {
  return rows.filter((r) => {
    if (batchPill && batchPill !== '全部' && r.batch !== Number(batchPill)) return false
    if (inLenPill && inLenPill !== '全部' && r.inLen !== Number(inLenPill)) return false
    return true
  })
}

export function inferPlatHasFilteredRow(
  platKey: string,
  inferTable: Record<string, { prefill?: { batch: number; inLen: number }[] } | undefined>,
  batchPill: string | undefined,
  inLenPill: string | undefined,
): boolean {
  const pre = inferTable[platKey]?.prefill || []
  return filterInferRows(pre, batchPill, inLenPill).length > 0
}
