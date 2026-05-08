/**
 * 训练维度 XLSX 业务规则（与 train 规格一致）。
 */

export type TrainTableRow = {
  /** 与 NVIDIA 对齐用：(framework, model, n_gpu, seq_len, dtype) */
  matchKey: string
  framework: string
  model: string
  parallel: string
  dtype: string
  /** 'on' 或原始 off 文案（含原因） */
  flashAttn: string
  tps: number
  baseline: number
  vsA100: number
  note: string
  nGpu: number
  seqLen: number
  microBatchSize: number
  /** XLSX date 原串，仅生成元数据用 */
  date?: string
}

export type TrainPersistRow = Omit<TrainTableRow, 'matchKey'>

export function trainMatchKey(
  framework: string,
  model: string,
  nGpu: number,
  seqLen: number,
  dtype: string,
): string {
  return [
    String(framework || '').trim().toLowerCase(),
    String(model || '').trim().toLowerCase(),
    String(nGpu),
    String(seqLen),
    String(dtype || '').trim().toLowerCase(),
  ].join('|')
}

/** 详情表「并行配置」：`{n_gpu} GPU · seq{seq_len}` */
export function formatTrainParallel(nGpu: number, seqLen: number): string {
  return `${nGpu} GPU · seq${seqLen}`
}

export function normalizeTrainDtype(dtype: string): string {
  return String(dtype || '').trim().toUpperCase()
}

/** zero_stage 非 default 时写入备注片段 */
export function buildTrainNote(zeroStage: string, remarks: string): string {
  const zs = String(zeroStage || '').trim()
  const rm = String(remarks || '').trim()
  const parts: string[] = []
  if (zs && zs.toLowerCase() !== 'default') parts.push(zs)
  if (rm) parts.push(rm)
  return parts.join(' · ')
}

export function parseFlashAttnCell(raw: string): string {
  const s = String(raw || '').trim()
  if (/^on$/i.test(s) || /^on[\s(]/i.test(s)) return 'on'
  return s || 'off'
}

export function trainVsPercent(platTps: number, nvTps: number): number {
  if (!Number.isFinite(platTps) || !Number.isFinite(nvTps) || nvTps <= 0) return 100
  return Math.round((platTps / nvTps) * 100)
}

export type TrainCardMetrics = {
  ownScore: number
  ownVal: string
  n: number
  extra: string
  adv: boolean
  advTxt: string
}

/**
 * 概览卡：代表行为吞吐最高行；ownScore 取该行 vs NVIDIA。
 */
export function buildTrainCardMetrics(rows: TrainTableRow[]): TrainCardMetrics | null {
  if (!rows.length) return null
  const best = rows.reduce((a, b) => (a.tps >= b.tps ? a : b))
  const ownScore = best.vsA100
  const ownVal = `${Math.round(best.tps)} tpps`
  const fw = best.framework.charAt(0).toUpperCase() + best.framework.slice(1).toLowerCase()
  const extra = `${fw} - ${best.model} - ${best.nGpu} GPU`
  const adv = ownScore >= 100
  const advTxt =
    ownScore >= 100
      ? `相对 NVIDIA 同配置 ${ownScore}%`
      : ownScore >= 70
        ? `相对 NVIDIA ${ownScore}%（可优化）`
        : `相对 NVIDIA ${ownScore}%`
  return {
    ownScore,
    ownVal,
    n: rows.length,
    extra,
    adv,
    advTxt,
  }
}

export function trainPlatHasFramework(
  platKey: string,
  trainTable: Record<string, { framework: string }[] | undefined>,
  frameworkPill: string | undefined,
): boolean {
  const rows = trainTable[platKey] || []
  if (!frameworkPill || frameworkPill === '全部') return rows.length > 0
  const want = frameworkPill.toLowerCase()
  return rows.some((r) => r.framework.toLowerCase() === want)
}

/** 顶栏「框架」pill 与训练详情表、图表联动 */
export function filterTrainRows<T extends { framework: string }>(
  rows: T[] | undefined,
  frameworkPill: string | undefined,
): T[] {
  const list = rows || []
  if (!frameworkPill || frameworkPill === '全部') return list
  const want = frameworkPill.toLowerCase()
  return list.filter((r) => r.framework.toLowerCase() === want)
}
