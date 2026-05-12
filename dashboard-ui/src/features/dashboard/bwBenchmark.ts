/**
 * 访存维度（`new_data/bw`）：文件列 `model`、`add_bw_GBps`、`copy_bw_GBps`、`scale_bw_GBps`、`triad_bw_GBps`、`bw_GBps`（表头规范化后为下划线小写）、`date`、`tester`、`remarks`。
 * 行级 vs NVIDIA：每行（每个 **model**）用该行 **bw_GBps（avg）÷ 1607.4561 × 100**；同一平台多型号各行独立对比。
 * 概览 / KPI：该平台 **MAX(bw_GBps)** 所在行（`pickBestBwRow`）。
 */

/** 规格图定值：NVIDIA A100 四模式均值基线（GB/s） */
export const BW_NVIDIA_BASELINE_GBPS = 1607.4561

export type BwDetailRow = {
  model: string
  add: number | null
  copy: number | null
  scale: number | null
  triad: number | null
  avg: number | null
  /** 行级：该行 bw_GBps（avg）÷ 1607.4561 × 100；按 model 每行独立 */
  vsNvidia: number
  remarks?: string
  tester?: string
  date?: string
}

export function bwVsNvidiaPercent(avgBw: number): number {
  if (!Number.isFinite(avgBw) || BW_NVIDIA_BASELINE_GBPS <= 0) return 100
  return Math.round((avgBw / BW_NVIDIA_BASELINE_GBPS) * 100)
}

export function pickBestBwRow(rows: BwDetailRow[]): BwDetailRow | null {
  const valid = rows.filter((r) => r.avg != null && Number.isFinite(r.avg))
  if (!valid.length) return null
  return valid.reduce((a, b) => ((a.avg ?? 0) >= (b.avg ?? 0) ? a : b))
}

export function buildBwCardMetrics(rows: BwDetailRow[]): {
  ownScore: number
  ownVal: string
  n: number
  extra: string
  adv: boolean
  advTxt: string
} | null {
  if (!rows.length) return null
  const best = pickBestBwRow(rows)
  if (!best || best.avg == null) return null
  const vs = bwVsNvidiaPercent(best.avg)
  const ownVal = `${best.avg.toFixed(1)} GB/s`
  const extra = best.model
  const adv = vs >= 100
  const advTxt = adv
    ? `HBM 均值相对 NVIDIA A100 基线 ${vs}%`
    : `相对 NVIDIA A100 基线 ${vs}%`
  return {
    ownScore: vs,
    ownVal,
    n: rows.length,
    extra,
    adv,
    advTxt,
  }
}

export function bwPlatHasMode(
  platKey: string,
  bwTable: Record<string, BwDetailRow[] | undefined>,
  modePill: string | undefined,
): boolean {
  const rows = bwTable[platKey] || []
  if (!modePill || modePill === '全部') return rows.length > 0
  const m = modePill.toLowerCase()
  if (m !== 'add' && m !== 'copy' && m !== 'scale' && m !== 'triad') return rows.length > 0
  const key = m as 'add' | 'copy' | 'scale' | 'triad'
  return rows.some((r) => {
    const v = r[key]
    return v != null && Number.isFinite(v)
  })
}

/** 当前平台访存行内 `date` 字典序最大 */
export function maxBwCsvDateForPlatform(
  bwTable: Record<string, { date?: string }[] | undefined>,
  platKey: string,
): string | undefined {
  let max = ''
  for (const r of bwTable[platKey] || []) {
    const d = String(r.date ?? '').trim()
    if (d && d > max) max = d
  }
  return max || undefined
}
