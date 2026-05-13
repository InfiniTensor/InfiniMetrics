/**
 * 通信维度业务规则（`new_data/comm`）。
 * 文件列 → 内存：`link_type`→linkType，`comm_type`→commType，`n_gpu`→nGpu，`bw_GBps`（表头规范化后为 `bw_gbps`）→bw，`remarks`→note，`date`→date。
 * 得分：`round(当前 bw ÷ NVIDIA 同 (comm_type, n_gpu) 基线 bw × 100)`，与详情表「vs NVIDIA」一致；NVIDIA 同键多行时基线取 **max(bw)**。
 */

export type CommImportRow = {
  matchKey: string
  linkType: string
  commType: string
  nGpu: number
  bw: number
  baseline: number
  vsA100: number
  /** 文件 `remarks` 列 */
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

/** 在 NVIDIA 表上按 (comm_type, n_gpu) 取用于对比的基线带宽（同键多行取 max，与 enrich 一致） */
export function commNvidiaBaselineBw(
  nvRows: { commType: string; nGpu: number; bw: number }[] | undefined,
  commType: string,
  nGpu: number,
): number | undefined {
  const want = normalizeCommType(commType)
  let max = 0
  let found = false
  for (const r of nvRows || []) {
    if (normalizeCommType(r.commType) !== want || r.nGpu !== nGpu) continue
    if (!Number.isFinite(r.bw)) continue
    found = true
    max = Math.max(max, r.bw)
  }
  return found && max > 0 ? max : undefined
}

/** 详情 / KPI / 卡片：带宽 GB/s 展示（整数无小数，否则一位小数） */
export function formatCommBandwidthGb(x: number): string {
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

function commPickMaxBwRow(rows: CommImportRow[], want: 'p2p' | 'allreduce'): CommImportRow | null {
  const list = rows.filter((r) => normalizeCommType(r.commType) === want)
  if (!list.length) return null
  return list.reduce((a, b) => (a.bw >= b.bw ? a : b))
}

/**
 * 概览卡：
 * - 子集仅含一种 comm_type（如顶栏筛成 p2p / allreduce）：大分 = 该类型下 **bw 最高** 一行的 `vsA100`，`openScore` 为空。
 * - 同时含 p2p 与 allreduce：左/右大分为各自类型 max-bw 行的 `vsA100`。
 */
export function buildCommCardMetrics(rows: CommImportRow[]): {
  ownScore: number
  openScore: number | null
  ownVal: string
  openVal: string | null
  n: number
  extra: string
  adv: boolean
  advTxt: string
  overviewOwnFw: string
  overviewOpenFw: string | null
} | null {
  if (!rows.length) return null
  const sorted = [...rows].sort((a, b) => {
    if (a.commType !== b.commType) return a.commType.localeCompare(b.commType)
    return a.nGpu - b.nGpu
  })
  const p2p = commPickMaxBwRow(sorted, 'p2p')
  const ar = commPickMaxBwRow(sorted, 'allreduce')
  if (!p2p && !ar) return null

  if (p2p && !ar) {
    const ownScore = p2p.vsA100
    const ownVal = `${formatCommBandwidthGb(p2p.bw)} GB/s`
    const linkRaw = String(p2p.linkType ?? '')
      .replace(/\s+/g, '')
      .toLowerCase()
    const extra = `${linkRaw || '—'} · ${p2p.nGpu}GPU`
    const adv = ownScore >= 100
    const advTxt =
      ownScore >= 100
        ? `P2P 相对 NVIDIA ${ownScore}%`
        : ownScore >= 70
          ? `P2P 相对 NVIDIA ${ownScore}%（可优化）`
          : `P2P 相对 NVIDIA ${ownScore}%`
    return {
      ownScore,
      openScore: null,
      ownVal,
      openVal: null,
      n: rows.length,
      extra,
      adv,
      advTxt,
      overviewOwnFw: 'p2p',
      overviewOpenFw: null,
    }
  }

  if (ar && !p2p) {
    const ownScore = ar.vsA100
    const ownVal = `${formatCommBandwidthGb(ar.bw)} GB/s`
    const linkRaw = String(ar.linkType ?? '')
      .replace(/\s+/g, '')
      .toLowerCase()
    const extra = `${linkRaw || '—'} · ${ar.nGpu}GPU`
    const adv = ownScore >= 100
    const advTxt =
      ownScore >= 100
        ? `AllReduce 相对 NVIDIA ${ownScore}%`
        : ownScore >= 70
          ? `AllReduce 相对 NVIDIA ${ownScore}%（可优化）`
          : `AllReduce 相对 NVIDIA ${ownScore}%`
    return {
      ownScore,
      openScore: null,
      ownVal,
      openVal: null,
      n: rows.length,
      extra,
      adv,
      advTxt,
      overviewOwnFw: 'allreduce',
      overviewOpenFw: null,
    }
  }

  const ownScore = p2p!.vsA100
  const openScore = ar!.vsA100
  const ownVal = `${formatCommBandwidthGb(p2p!.bw)} GB/s`
  const openVal = `${formatCommBandwidthGb(ar!.bw)} GB/s`
  const rep = (p2p!.bw >= ar!.bw ? p2p! : ar!)
  const linkRaw = String(rep.linkType ?? '')
    .replace(/\s+/g, '')
    .toLowerCase()
  const extra = `${linkRaw || '—'} · ${rep.nGpu}GPU`

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
    overviewOwnFw: 'p2p',
    overviewOpenFw: 'allreduce',
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

/** 当前平台通信行内 `date` 字典序最大（与侧栏「数据更新于」一致） */
export function maxCommCsvDateForPlatform(
  commTable: Record<string, { date?: string }[] | undefined>,
  platKey: string,
): string | undefined {
  let max = ''
  for (const r of commTable[platKey] || []) {
    const d = String(r.date ?? '').trim()
    if (d && d > max) max = d
  }
  return max || undefined
}
