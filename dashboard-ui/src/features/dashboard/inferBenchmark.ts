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
  /** 原始 CSV date，横条可选展示 */
  date?: string
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
  date?: string
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

/**
 * 行级 vs NVIDIA：优先同 configKey；否则同 batch+input_tokens 下先同 output_tokens，再取 NVIDIA 同组 TPS 最大。
 */
export function enrichInferWithNvidiaBaseline(
  prefill: InferPrefillRow[],
  decode: InferDecodeRow[],
  nvPrefill: InferPrefillRow[],
  nvDecode: InferDecodeRow[],
): void {
  function nvPrefillBaseline(
    row: Pick<InferPrefillRow, 'configKey' | 'batch' | 'inLen' | 'outLen'>,
  ): number | null {
    const exact = nvPrefill.find((r) => r.configKey === row.configKey)
    if (exact) return exact.tps
    const cand = nvPrefill.filter((r) => r.batch === row.batch && r.inLen === row.inLen)
    if (!cand.length) return null
    const sameOut = cand.find((r) => r.outLen === row.outLen)
    if (sameOut) return sameOut.tps
    return Math.max(...cand.map((r) => r.tps))
  }
  function nvDecodeBaseline(
    row: Pick<InferDecodeRow, 'configKey' | 'batch' | 'inLen' | 'outLen'>,
  ): number | null {
    const exact = nvDecode.find((r) => r.configKey === row.configKey)
    if (exact) return exact.tps
    const cand = nvDecode.filter((r) => r.batch === row.batch && r.inLen === row.inLen)
    if (!cand.length) return null
    const sameOut = cand.find((r) => r.outLen === row.outLen)
    if (sameOut) return sameOut.tps
    return Math.max(...cand.map((r) => r.tps))
  }
  for (const row of prefill) {
    const nb = nvPrefillBaseline(row)
    row.nvidiaBaselineTps = nb ?? null
    row.vsNvidia = nb != null && nb > 0 ? inferVsPercent(row.tps, nb) : null
  }
  for (const row of decode) {
    const nb = nvDecodeBaseline(row)
    row.nvidiaBaselineTps = nb ?? null
    row.vsNvidia = nb != null && nb > 0 ? inferVsPercent(row.tps, nb) : null
  }
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
  /** 概览「配置」：Prefill 峰值所在行的 batch + input_tokens（及可选 Decode 峰值行） */
  extra: string
  /** 左列大分下方小字 */
  inferOwnCaption?: string
  /** 右列大分下方小字 */
  inferOpenCaption?: string
  adv: boolean
  advTxt: string
  dataDate?: string
}

/**
 * 概览卡：Prefill/Decode 峰值与 NVIDIA 全表最大值的百分比；extra 记录峰值行 batch+in。
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
  let argD: InferFlatRow | null = null
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
      if (maxD == null || r.decodeTps > maxD) {
        maxD = r.decodeTps
        argD = r
      }
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

  const pBi = argP ? `batch=${argP.batch} in=${argP.inLen}` : ''
  const dBi = argD ? `batch=${argD.batch} in=${argD.inLen}` : ''
  const extra =
    pBi && dBi && (argP!.batch !== argD!.batch || argP!.inLen !== argD!.inLen)
      ? `Prefill ${pBi} · Decode ${dBi}`
      : pBi || dBi || ''

  const adv = ownScore >= 100
  const advTxt =
    ownScore >= 100
      ? `Prefill 相对 NVIDIA ${ownScore}%`
      : ownScore >= 70
        ? `Prefill 相对 NVIDIA ${ownScore}%（可优化）`
        : `Prefill 相对 NVIDIA ${ownScore}%`

  return {
    ownScore,
    openScore,
    ownVal,
    openVal,
    n,
    extra,
    inferOwnCaption: maxP != null ? 'Prefill 最优' : undefined,
    inferOpenCaption: maxD != null ? 'Decode 最优' : undefined,
    adv,
    advTxt,
    dataDate,
  }
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

export function filterInferRows<T extends { batch: number; inLen: number }>(
  rows: T[],
  batchPill: string | undefined,
  inLenPill: string | undefined,
): T[] {
  return rows.filter((r) => {
    if (batchPill && batchPill !== '全部' && r.batch !== Number(batchPill)) return false
    if (inLenPill && inLenPill !== '全部' && r.inLen !== Number(inLenPill)) return false
    return true
  })
}

/** 推理表：用于筛选项并集 / 按平台子集 */
export type InferTablePack = {
  prefill?: { batch: number; inLen: number; dtype?: string; date?: string }[]
  decode?: { batch: number; inLen: number; dtype?: string; date?: string }[]
}

function inferRowsBothTabs(pack: InferTablePack | undefined) {
  return [...(pack?.prefill ?? []), ...(pack?.decode ?? [])]
}

export function inferNumericSetForPlatform(
  tbl: Record<string, InferTablePack | undefined>,
  platKey: string,
  field: 'batch' | 'inLen',
): Set<number> {
  const set = new Set<number>()
  for (const r of inferRowsBothTabs(tbl[platKey])) {
    const v = r[field]
    if (Number.isFinite(v)) set.add(v)
  }
  return set
}

export function inferBatchPillsUnion(tbl: Record<string, InferTablePack | undefined>): string[] {
  const s = new Set<number>()
  for (const pk of Object.keys(tbl)) {
    for (const n of inferNumericSetForPlatform(tbl, pk, 'batch')) s.add(n)
  }
  return [...s].sort((a, b) => a - b).map(String)
}

export function inferInLenPillsUnion(tbl: Record<string, InferTablePack | undefined>): string[] {
  const s = new Set<number>()
  for (const pk of Object.keys(tbl)) {
    for (const n of inferNumericSetForPlatform(tbl, pk, 'inLen')) s.add(n)
  }
  return [...s].sort((a, b) => a - b).map(String)
}

/** 当前平台 prefill ∪ decode 行内 `date` 字典序最大（CSV date） */
export function maxInferCsvDateForPlatform(
  tbl: Record<string, InferTablePack | undefined>,
  platKey: string,
): string | undefined {
  let max = ''
  for (const r of inferRowsBothTabs(tbl[platKey])) {
    const d = String(r.date ?? '').trim()
    if (d && d > max) max = d
  }
  return max || undefined
}

export function inferPlatHasFilteredRow(
  platKey: string,
  inferTable: Record<string, InferTablePack | undefined>,
  batchPill: string | undefined,
  inLenPill: string | undefined,
): boolean {
  const pack = inferTable[platKey]
  const pre = pack?.prefill || []
  const dec = pack?.decode || []
  return (
    filterInferRows(pre, batchPill, inLenPill).length > 0 ||
    filterInferRows(dec, batchPill, inLenPill).length > 0
  )
}
