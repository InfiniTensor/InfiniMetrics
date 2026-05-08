/**
 * 通信维度 XLSX 业务规则（与 comm 规格一致）。
 * 行级 vs NVIDIA：match (comm_type, n_gpu)；基线为 NVIDIA 同键 bw_GBps。
 */

export type CommImportRow = {
  matchKey: string
  linkType: string
  commType: string
  nGpu: number
  bw: number
  baseline: number
  vsA100: number
  note: string
  date?: string
}

export function commMatchKey(commType: string, nGpu: number): string {
  return `${String(commType || '').trim().toLowerCase()}|${nGpu}`
}

export function normalizeCommType(raw: string): string {
  return String(raw || '')
    .trim()
    .toLowerCase()
    .replace(/\s+/g, '')
}

/** 详情表「Link 类型」展示 */
export function formatCommLinkType(raw: string): string {
  const s = String(raw || '').trim().toLowerCase()
  if (!s) return '—'
  if (s === 'nvlink') return 'NVLink'
  if (s === 'metaxlink') return 'MetaxLink'
  return s.charAt(0).toUpperCase() + s.slice(1)
}

export function commVsPercent(platBw: number, nvBw: number): number {
  if (!Number.isFinite(platBw) || !Number.isFinite(nvBw) || nvBw <= 0) return 100
  return Math.round((platBw / nvBw) * 100)
}

function formatBwGb(x: number): string {
  if (!Number.isFinite(x)) return '—'
  return Number.isInteger(x) ? String(Math.round(x)) : x.toFixed(1)
}

export function enrichCommBaselines(byPlat: Record<string, CommImportRow[]>) {
  const nv = byPlat.nvidia || []
  const baselineMap = new Map<string, number>()
  for (const r of nv) {
    const prev = baselineMap.get(r.matchKey) ?? 0
    baselineMap.set(r.matchKey, Math.max(prev, r.bw))
  }
  for (const plat of Object.keys(byPlat)) {
    for (const row of byPlat[plat]) {
      const b = baselineMap.get(row.matchKey)
      if (b != null && b > 0) {
        row.baseline = b
        row.vsA100 = commVsPercent(row.bw, b)
      } else {
        row.baseline = row.bw
        row.vsA100 = 100
      }
    }
  }
}

/**
 * 概览卡：ownVal / openVal 为 P2P / AllReduce 带宽；ownScore / openScore 为各自 vs NVIDIA。
 */
export function buildCommCardMetrics(rows: CommImportRow[]): {
  ownScore: number
  openScore: number
  ownVal: string
  openVal: string
  n: number
  extra: string
  adv: boolean
  advTxt: string
} | null {
  if (!rows.length) return null
  const sorted = [...rows].sort((a, b) => {
    if (a.commType !== b.commType) return a.commType.localeCompare(b.commType)
    return a.nGpu - b.nGpu
  })
  const p2p = sorted.find((r) => r.commType === 'p2p')
  const ar = sorted.find((r) => r.commType === 'allreduce')
  if (!p2p && !ar) return null

  const ownScore = p2p?.vsA100 ?? 100
  const openScore = ar?.vsA100 ?? 100
  const ownVal = p2p ? `${formatBwGb(p2p.bw)} GB/s` : '—'
  const openVal = ar ? `${formatBwGb(ar.bw)} GB/s` : '—'
  const extra = (p2p || ar)!.linkType

  const adv = ownScore >= 100 || openScore >= 100
  let advTxt = `P2P ${ownScore}% · AllReduce ${openScore}%（相对 NVIDIA）`
  if (ownScore >= 100 && openScore >= 100) advTxt = 'P2P / AllReduce 均达或超过 NVIDIA 基线'
  else if (openScore >= 100 && ownScore < 100)
    advTxt = `AllReduce 相对 NVIDIA ${openScore}%（P2P ${ownScore}%）`
  else if (ownScore >= 100 && openScore < 100)
    advTxt = `P2P 相对 NVIDIA ${ownScore}%（AllReduce ${openScore}%）`

  return {
    ownScore,
    openScore,
    ownVal,
    openVal,
    n: rows.length,
    extra,
    adv,
    advTxt,
  }
}

export function commPlatHasCommType(
  platKey: string,
  commTable: Record<string, { commType: string }[] | undefined>,
  commPill: string | undefined,
): boolean {
  const rows = commTable[platKey] || []
  if (!commPill || commPill === '全部') return rows.length > 0
  const want = normalizeCommType(commPill)
  return rows.some((r) => normalizeCommType(r.commType) === want)
}

export function filterCommRows<T extends { commType: string }>(
  rows: T[] | undefined,
  commTypePill: string | undefined,
): T[] {
  const list = rows || []
  if (!commTypePill || commTypePill === '全部') return list
  const want = normalizeCommType(commTypePill)
  return list.filter((r) => normalizeCommType(r.commType) === want)
}
