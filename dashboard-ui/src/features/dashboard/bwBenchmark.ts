/**
 * 访存维度 CSV 业务规则：行级 vs NVIDIA 分母为 NVIDIA A100 HBM 均值常数；
 * 概览卡 / KPI 取该平台 MAX(bw_GBps) 所在行（最佳型号）。
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
  /** bw_GBps ÷ 1607.4561 × 100，avg 为空时可置 100 */
  vsNvidia: number
  remarks?: string
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
  const vs = best.vsNvidia
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
