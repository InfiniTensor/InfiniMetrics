/**
 * 访存维度（`new_data/bw`）：文件列 `model`、`add_bw_GBps`、`copy_bw_GBps`、`scale_bw_GBps`、`triad_bw_GBps`、`bw_GBps`（表头规范化后为下划线小写）、`date`、`tester`、`remarks`。
 * 得分统一：当前指标 ÷ **NVIDIA A100 固定基线** `BW_NVIDIA_BASELINE_GBPS` × 100（与详情「vs NVIDIA」一致）。
 * 行级：四模式均值 `avg` 对 A100 得 `vsNvidia`；各模式单列亦可同公式对比 A100。
 * 概览「全部」：`pickBestBwRow`（按 avg）行的 `vsNvidia`；extra 为全表 model 去重拼接。
 */

/** 规格：NVIDIA A100 四模式均值对标基线（GB/s） */
export const BW_NVIDIA_BASELINE_GBPS = 1607.46

export type BwDetailRow = {
  model: string
  add: number | null
  copy: number | null
  scale: number | null
  triad: number | null
  avg: number | null
  /** 行级：该行四模式均值 `avg` 对 A100 基线之百分比（与 `bwVsNvidiaPercent(avg)` 一致） */
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

/** 访存单模式列（add/copy/scale/triad）；`mode` 为空时与 `pickBestBwRow`（按 avg）一致 */
export type BwModeKey = 'add' | 'copy' | 'scale' | 'triad'

export function pickBestBwRowByMode(
  rows: BwDetailRow[],
  mode: BwModeKey | null | undefined,
): BwDetailRow | null {
  if (!mode) return pickBestBwRow(rows)
  const mk = mode
  const withV = rows.filter((r) => {
    const v = r[mk]
    return v != null && Number.isFinite(v)
  })
  if (!withV.length) return null
  return withV.reduce((a, b) => ((a[mk] ?? 0) >= (b[mk] ?? 0) ? a : b))
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
  /** 大分与详情表「vs NVIDIA」列一致（avg 对 A100） */
  const vs = best.vsNvidia
  const ownVal = `${best.avg.toFixed(1)} GB/s`
  const names = [...new Set(rows.map((r) => String(r.model || '').trim()).filter(Boolean))].sort((a, b) =>
    a.localeCompare(b, undefined, { sensitivity: 'base' }),
  )
  const extra = names.length ? names.join(' / ') : best.model || '—'
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
