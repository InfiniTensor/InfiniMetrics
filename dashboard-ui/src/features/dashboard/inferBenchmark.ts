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
  /** 概览「配置」：Prefill / Decode 各自峰值相对 NVIDIA 的百分数较高者所在行的 batch · in-len · out-len；平局取 Prefill 行 */
  extra: string
  /** 左列大分下方小字 */
  inferOwnCaption?: string
  /** 右列大分下方小字 */
  inferOpenCaption?: string
  adv: boolean
  advTxt: string
  dataDate?: string
}

/** 与 import 脚本一致：扁平行上取 Prefill / Decode TPS 峰值（供概览按筛选项重算分母） */
export function maxInferMetric(rows: InferFlatRow[], key: 'prefillTps' | 'decodeTps'): number {
  let m = 0
  for (const r of rows) {
    const v = r[key]
    if (v != null && Number.isFinite(v) && v > m) m = v
  }
  return m
}

/**
 * 由 INFER_TABLE 的 prefill / decode 拼回与 `parseInferCsvNew` 一致的 InferFlatRow[]，
 * 供概览在切换 Batch / In-len 后按 `buildInferCardMetrics` 重算卡片。
 */
export function inferFlatsFromTablePack(
  platKey: string,
  pack: InferTablePack | undefined,
): InferFlatRow[] {
  if (!pack) return []
  /** 静态 INFER_TABLE 与 CSV 导入行字段超集（部分行无 configKey） */
  type PackRow = {
    configKey?: string
    batch: number
    inLen: number
    outLen?: number
    model?: string
    dtype?: string
    nGpu?: number
    remarks?: string
    date?: string
    tps?: number
    ttft?: number
    decodeLatencyMs?: number
  }
  const prefill = (pack.prefill ?? []) as PackRow[]
  const decodeRows = (pack.decode ?? []) as PackRow[]
  const decByKey = new Map(
    decodeRows.map((r) => [r.configKey ?? inferConfigKey(r.batch, r.inLen, r.outLen ?? 0), r]),
  )
  const seen = new Set<string>()
  const flats: InferFlatRow[] = []

  for (const pr of prefill) {
    const ck = pr.configKey ?? inferConfigKey(pr.batch, pr.inLen, pr.outLen ?? 0)
    seen.add(ck)
    const dr = decByKey.get(ck)
    const prefillTps = pr.tps != null && Number.isFinite(pr.tps) ? pr.tps : null
    const decodeTps = dr?.tps != null && Number.isFinite(dr.tps) ? dr.tps : null
    if (prefillTps == null && decodeTps == null) continue
    flats.push({
      plat: platKey,
      batch: pr.batch,
      inLen: pr.inLen,
      outLen: pr.outLen ?? dr?.outLen ?? 0,
      model: normalizeInferModel(pr.model ?? ''),
      dtype: String(pr.dtype || '').trim().toUpperCase(),
      nGpu: pr.nGpu ?? 0,
      remarks: String(pr.remarks ?? '').trim(),
      date: String(pr.date ?? '').trim(),
      ttftMs: pr.ttft != null && Number.isFinite(pr.ttft) && pr.ttft > 0 ? pr.ttft : null,
      decodeMs:
        pr.decodeLatencyMs != null &&
        Number.isFinite(pr.decodeLatencyMs) &&
        pr.decodeLatencyMs > 0
          ? pr.decodeLatencyMs
          : null,
      prefillTps,
      decodeTps,
    })
  }

  for (const dr of decodeRows) {
    const ck = dr.configKey ?? inferConfigKey(dr.batch, dr.inLen, dr.outLen ?? 0)
    if (seen.has(ck)) continue
    const decodeTps = dr.tps != null && Number.isFinite(dr.tps) ? dr.tps : null
    if (decodeTps == null) continue
    flats.push({
      plat: platKey,
      batch: dr.batch,
      inLen: dr.inLen,
      outLen: dr.outLen ?? 0,
      model: normalizeInferModel(dr.model ?? ''),
      dtype: String(dr.dtype || '').trim().toUpperCase(),
      nGpu: dr.nGpu ?? 0,
      remarks: String(dr.remarks ?? '').trim(),
      date: String(dr.date ?? '').trim(),
      ttftMs: null,
      decodeMs:
        dr.decodeLatencyMs != null &&
        Number.isFinite(dr.decodeLatencyMs) &&
        dr.decodeLatencyMs > 0
          ? dr.decodeLatencyMs
          : null,
      prefillTps: null,
      decodeTps,
    })
  }

  flats.sort((a, b) => {
    if (a.batch !== b.batch) return a.batch - b.batch
    if (a.inLen !== b.inLen) return a.inLen - b.inLen
    return a.outLen - b.outLen
  })
  return flats
}

/**
 * 概览卡：Prefill / Decode 得分取各行 vs NVIDIA（同详情表）的最大值；
 * 吞吐展示对应取得分最高的那一行；extra 为两路得分较高者所在行的 batch · in-len · out-len（平局取 Prefill）。
 */
export type InferOverviewMetricRow = {
  batch: number
  inLen: number
  outLen?: number
  tps: number
  vsNvidia?: number | null
  date?: string
}

function pickMaxVsNvidiaRow<T extends InferOverviewMetricRow>(rows: T[]): T | null {
  let best: T | null = null
  let maxVs = -Infinity
  for (const r of rows) {
    const v = r.vsNvidia
    if (v == null || !Number.isFinite(v)) continue
    if (best == null || v > maxVs) {
      maxVs = v
      best = r
    }
  }
  return best
}

export function buildInferCardMetrics(
  prefillRows: InferOverviewMetricRow[],
  decodeRows: InferOverviewMetricRow[],
): InferCardMetrics | null {
  if (!prefillRows.length && !decodeRows.length) return null
  const argP = pickMaxVsNvidiaRow(prefillRows)
  const argD = pickMaxVsNvidiaRow(decodeRows)
  const dates: string[] = []
  for (const r of [...prefillRows, ...decodeRows]) {
    if (r.date) dates.push(r.date)
  }
  const dataDate = dates.length ? [...dates].sort().at(-1) : undefined

  const ownScore = argP?.vsNvidia ?? 0
  const openScore = argD?.vsNvidia ?? 0
  const ownVal = argP != null ? formatInferTokPerS(argP.tps) : '—'
  const openVal = argD != null ? formatInferTokPerS(argD.tps) : '—'
  const n = prefillRows.length + decodeRows.length

  const arg =
    argP && argD ? (ownScore >= openScore ? argP : argD) : (argP ?? argD)
  const extra = arg
    ? `batch=${arg.batch} in=${arg.inLen} out=${arg.outLen ?? 0}`
    : ''

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
    inferOwnCaption: argP != null ? 'Prefill 最优' : undefined,
    inferOpenCaption: argD != null ? 'Decode 最优' : undefined,
    adv,
    advTxt,
    dataDate,
  }
}

/**
 * 详情吞吐柱图：按 batch+inLen 归并（同一组多 out_len 取各侧 TPS 最大），
 * 类目 `bs{batch} in{inLen}`，与产品 X 轴格式一致。
 * 仅保留「当前平台与 NVIDIA 在该 (batch, inLen) 上均有有效吞吐」的类目，避免单侧无数据仍占轴位。
 */
export function alignInferBarByBatchIn<
  T extends { batch: number; inLen: number; tps: number },
>(plat: T[], nv: T[]): { categories: string[]; platVals: number[]; nvVals: number[] } {
  const biKey = (batch: number, inLen: number) => `${batch}:${inLen}`
  function mergeMax(rows: T[]) {
    const m = new Map<string, number>()
    for (const r of rows) {
      if (!Number.isFinite(r.tps)) continue
      const k = biKey(r.batch, r.inLen)
      m.set(k, Math.max(m.get(k) ?? 0, r.tps))
    }
    return m
  }
  const pm = mergeMax(plat)
  const nm = mergeMax(nv)
  const keys = new Set<string>()
  for (const k of pm.keys()) {
    if (nm.has(k)) keys.add(k)
  }
  const sorted = [...keys].sort((a, b) => {
    const [ab, ai] = a.split(':').map(Number)
    const [bb, bi] = b.split(':').map(Number)
    if (ab !== bb) return ab - bb
    return ai - bi
  })
  return {
    categories: sorted.map((k) => {
      const [b, i] = k.split(':')
      return `bs${b} in${i}`
    }),
    platVals: sorted.map((k) => pm.get(k) ?? 0),
    nvVals: sorted.map((k) => nm.get(k) ?? 0),
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

/** 推理表：用于筛选项并集 / 按平台子集（行字段超集，含详情表 vsNvidia） */
export type InferTablePack = {
  prefill?: (InferOverviewMetricRow & { dtype?: string; [key: string]: unknown })[]
  decode?: (InferOverviewMetricRow & { dtype?: string; [key: string]: unknown })[]
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
