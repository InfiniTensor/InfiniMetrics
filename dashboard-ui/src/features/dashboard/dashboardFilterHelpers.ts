import {
  DIMS,
  INFER_TABLE,
  OP_TABLE,
  TRAIN_TABLE,
  COMM_TABLE,
  BW_TABLE,
  type CommDetailRow,
  type TrainDetailRow,
} from '@/data'
import {
  buildBwCardMetrics,
  bwPlatHasMode,
  bwVsNvidiaPercent,
  pickBestBwRow,
  type BwDetailRow,
} from '@/features/dashboard/bwBenchmark'
import {
  buildCommCardMetrics,
  commPlatHasCommType,
  filterCommRows,
  type CommImportRow,
} from '@/features/dashboard/commBenchmark'
import {
  buildInferCardMetrics,
  filterInferRows,
  inferFlatsFromTablePack,
  inferPlatHasFilteredRow,
  maxInferMetric,
  type InferFlatRow,
  type InferTablePack,
} from '@/features/dashboard/inferBenchmark'
import {
  buildTrainCardMetrics,
  filterTrainRows,
  trainPlatHasFramework,
  type TrainTableRow,
} from '@/features/dashboard/trainBenchmark'
import {
  buildOpPlatformOverview,
  canComputeOpRowScore,
  computeOpRowScore,
} from '@/features/dashboard/operatorBenchmark'

/** 卡片行（与各维度 CARD_DATA 元素一致的最小字段） */
export type CardRow = {
  key: string
  /** 算子维度：详情「测试记录」副标题，如「7 算子」（概览「配置」用 extra） */
  opRecordSub?: string
  extra?: string
  ownScore?: number | null
  ownVal?: string | null
  openVal?: string | null
  openScore?: number | null
  n?: number
  ownFw?: string
  openFw?: string
  /** 通信概览双列时：底部「得分」= max(p2p vs%, allreduce vs%)；其它维度或未设时底部仍用 ownScore */
  footerScore?: number | null
  /** 推理概览：左/右列大分下方小字 */
  inferOwnCaption?: string
  inferOpenCaption?: string
  adv?: boolean
  advTxt?: string
}

export type OpTableForOverlay = Record<
  string,
  Record<string, { shape: string; dtype: string; ic: number; pt: number; remarks?: string; date?: string }[]>
>

function normOpDtype(v: unknown): string {
  return String(v ?? '').trim().toUpperCase()
}

/** 按顶栏算子类型 / 精度筛选项，从 OP_TABLE 拉平为 buildOpPlatformOverview 输入 */
export function collectOpFlatRowsForPills(
  platKey: string,
  tbl: OpTableForOverlay,
  opPill: string,
  dtypePill: string,
): Array<{
  opKey: string
  shape: string
  dtype: string
  ic: number
  pt: number
  remarks: string
  date?: string
}> {
  const platOps = tbl[platKey]
  if (!platOps) return []
  const wantD = dtypePill !== '全部' ? normOpDtype(dtypePill) : ''
  const out: Array<{
    opKey: string
    shape: string
    dtype: string
    ic: number
    pt: number
    remarks: string
    date?: string
  }> = []
  for (const [opKey, rows] of Object.entries(platOps)) {
    if (!Array.isArray(rows)) continue
    if (opPill !== '全部' && opKey !== opPill) continue
    for (const row of rows) {
      if (wantD && normOpDtype(row.dtype) !== wantD) continue
      out.push({
        opKey,
        shape: row.shape,
        dtype: row.dtype,
        ic: row.ic,
        pt: row.pt,
        remarks: row.remarks ?? '',
        date: row.date,
      })
    }
  }
  return out
}

/**
 * 概览页：在「算子类型 / 精度」非「全部」时，用 OP_TABLE 子集重算卡片 KPI（与 buildOpPlatformOverview 口径一致）。
 * 无匹配行时保留卡片，展示「无测试数据」。
 */
export function overlayOpOverviewCardFromFilters(
  card: CardRow,
  tbl: OpTableForOverlay,
  opPill: string,
  dtypePill: string,
): CardRow {
  const flat = collectOpFlatRowsForPills(card.key, tbl, opPill, dtypePill)
  if (!flat.length) {
    const scope = [opPill !== '全部' ? opPill : null, dtypePill !== '全部' ? dtypePill : null]
      .filter(Boolean)
      .join(' · ')
    return {
      ...card,
      ownScore: null,
      openScore: null,
      ownVal: '—',
      openVal: '—',
      n: 0,
      extra: scope ? `${scope} · 无测试数据` : '无测试数据',
      opRecordSub: '0 算子',
      adv: false,
      /** 产品要求：概览卡禁止再展示「暂无该筛选数据」之类弱化文案，留空让 a-tag 不渲染 */
      advTxt: '',
    }
  }
  const built = buildOpPlatformOverview(flat)!
  return {
    ...card,
    ownScore: built.ownScore,
    openScore: card.openScore ?? 100,
    ownVal: built.ownVal,
    openVal: built.openVal,
    n: built.n,
    extra: built.extra,
    opRecordSub: built.opRecordSub,
    adv: built.adv,
    advTxt: built.advTxt,
  }
}

function overviewCardNoFilterData(card: CardRow, scope: string): CardRow {
  return {
    ...card,
    ownScore: null,
    openScore: null,
    ownVal: '—',
    openVal: '—',
    n: 0,
    extra: scope ? `${scope} · 无测试数据` : '无测试数据',
    opRecordSub: card.opRecordSub,
    inferOwnCaption: undefined,
    inferOpenCaption: undefined,
    adv: false,
    /** 产品要求：概览卡禁止再展示「暂无该筛选数据」之类弱化文案，留空让 a-tag 不渲染 */
    advTxt: '',
    footerScore: null,
  }
}

/**
 * 推理概览：Batch / In-len 非「全部」时按 INFER_TABLE 子集重算（与 `buildInferCardMetrics` / import 脚本一致）。
 */
export function overlayInferOverviewCardFromFilters(
  card: CardRow,
  inferTbl: Record<string, InferTablePack | undefined>,
  batchPill: string | undefined,
  inLenPill: string | undefined,
): CardRow {
  const scope = [
    batchPill && batchPill !== '全部' ? `batch=${batchPill}` : null,
    inLenPill && inLenPill !== '全部' ? `in=${inLenPill}` : null,
  ]
    .filter(Boolean)
    .join(' ')
  let nvFlats = filterInferRows(
    inferFlatsFromTablePack('nvidia', inferTbl.nvidia),
    batchPill,
    inLenPill,
  ) as InferFlatRow[]
  let nvMaxP = maxInferMetric(nvFlats, 'prefillTps')
  let nvMaxD = maxInferMetric(nvFlats, 'decodeTps')
  if (nvMaxP <= 0 || nvMaxD <= 0) {
    for (const pk of Object.keys(inferTbl)) {
      const f = filterInferRows(
        inferFlatsFromTablePack(pk, inferTbl[pk]),
        batchPill,
        inLenPill,
      ) as InferFlatRow[]
      nvMaxP = Math.max(nvMaxP, maxInferMetric(f, 'prefillTps'))
      nvMaxD = Math.max(nvMaxD, maxInferMetric(f, 'decodeTps'))
    }
  }
  const platFlats = filterInferRows(
    inferFlatsFromTablePack(card.key, inferTbl[card.key]),
    batchPill,
    inLenPill,
  ) as InferFlatRow[]
  const m = buildInferCardMetrics(platFlats, nvMaxP || 1, nvMaxD || 1)
  if (!m) {
    return overviewCardNoFilterData(card, scope)
  }
  return {
    ...card,
    ownFw: card.ownFw ?? 'Prefill ✦',
    openFw: card.openFw ?? 'Decode',
    ownScore: m.ownScore,
    openScore: m.openScore,
    ownVal: m.ownVal,
    openVal: m.openVal,
    n: m.n,
    extra: m.extra,
    inferOwnCaption: m.inferOwnCaption,
    inferOpenCaption: m.inferOpenCaption,
    adv: m.adv,
    advTxt: m.advTxt,
  }
}

/** 训练概览：框架非「全部」时按 TRAIN_TABLE 子集重算 */
export function overlayTrainOverviewCardFromFilters(
  card: CardRow,
  trainTbl: Record<string, TrainDetailRow[] | undefined>,
  frameworkPill: string | undefined,
): CardRow {
  const rows = filterTrainRows(trainTbl[card.key], frameworkPill)
  const m = buildTrainCardMetrics(rows as unknown as TrainTableRow[])
  if (!m) {
    const scope =
      frameworkPill && frameworkPill !== '全部' ? frameworkPill : ''
    return overviewCardNoFilterData(card, scope)
  }
  return {
    ...card,
    openScore: null,
    openVal: null,
    ownScore: m.ownScore,
    ownVal: m.ownVal,
    n: m.n,
    extra: m.extra,
    ownFw: m.ownFwLabel || card.ownFw,
    openFw: card.openFw ?? '',
    adv: m.adv,
    advTxt: m.advTxt,
  }
}

/** 通信概览：通信类型非「全部」时按 COMM_TABLE 子集重算 */
export function overlayCommOverviewCardFromFilters(
  card: CardRow,
  commTbl: Record<string, CommDetailRow[] | undefined>,
  typePill: string | undefined,
): CardRow {
  const rows = filterCommRows(commTbl[card.key], typePill)
  const m = buildCommCardMetrics(rows as unknown as CommImportRow[])
  if (!m) {
    const scope = typePill && typePill !== '全部' ? typePill : ''
    return overviewCardNoFilterData(card, scope)
  }
  return {
    ...card,
    ownFw: m.overviewOwnFw,
    openFw: m.overviewOpenFw ?? '',
    ownScore: m.ownScore,
    openScore: m.openScore,
    ownVal: m.ownVal,
    openVal: m.openVal,
    n: m.n,
    extra: m.extra,
    adv: m.adv,
    advTxt: m.advTxt,
    footerScore: m.footerScore,
  }
}

type BwModeKey = 'add' | 'copy' | 'scale' | 'triad'

/** 访存概览：模式非「全部」时按该模式列峰值重算；「全部」与 `buildBwCardMetrics` 一致 */
export function overlayBwOverviewCardFromFilters(
  card: CardRow,
  bwTbl: Record<string, BwDetailRow[] | undefined>,
  modePill: string | undefined,
): CardRow {
  const rows = bwTbl[card.key] || []
  if (!modePill || modePill === '全部') {
    const m = buildBwCardMetrics(rows)
    if (!m) return overviewCardNoFilterData(card, '访存')
    return {
      ...card,
      ownFw: card.ownFw ?? 'HBM均值',
      openFw: card.openFw ?? '',
      openScore: null,
      openVal: null,
      ownScore: m.ownScore,
      ownVal: m.ownVal,
      n: m.n,
      extra: m.extra,
      adv: m.adv,
      advTxt: m.advTxt,
    }
  }
  const mk = modePill.toLowerCase() as BwModeKey
  if (mk !== 'add' && mk !== 'copy' && mk !== 'scale' && mk !== 'triad') {
    const m = buildBwCardMetrics(rows)
    if (!m) return overviewCardNoFilterData(card, modePill)
    return {
      ...card,
      ownFw: card.ownFw ?? 'HBM均值',
      openFw: card.openFw ?? '',
      openScore: null,
      openVal: null,
      ownScore: m.ownScore,
      ownVal: m.ownVal,
      n: m.n,
      extra: m.extra,
      adv: m.adv,
      advTxt: m.advTxt,
    }
  }
  const withMode = rows.filter((r) => {
    const v = r[mk]
    return v != null && Number.isFinite(v)
  })
  if (!withMode.length) {
    return overviewCardNoFilterData(card, modePill)
  }
  const best = withMode.reduce((a, b) => ((a[mk] ?? 0) >= (b[mk] ?? 0) ? a : b))
  const val = best[mk] as number
  const nvRows = bwTbl.nvidia || []
  const nvRef = pickBestBwRow(nvRows) ?? nvRows[0] ?? null
  const nvV = nvRef?.[mk]
  const vs =
    nvV != null && Number.isFinite(nvV) && nvV > 0
      ? Math.round((val / nvV) * 100)
      : bwVsNvidiaPercent(val)
  const adv = vs >= 100
  const advTxt = adv
    ? `${modePill} 带宽相对 NVIDIA 参考行同模式 ${vs}%`
    : `${modePill} 相对 NVIDIA 参考行同模式 ${vs}%`
  return {
    ...card,
    ownFw: card.ownFw ?? 'HBM均值',
    openFw: card.openFw ?? '',
    openScore: null,
    openVal: null,
    ownScore: vs,
    ownVal: `${val.toFixed(1)} GB/s`,
    n: withMode.length,
    extra: String(best.model || '').trim() || '—',
    adv,
    advTxt,
  }
}

/** 与 HTML `applyCardFilter` 行为一致（概览算子维度由 overlayOpOverviewCardFromFilters 另行重算指标） */
export function applyCardFilter(
  cards: CardRow[],
  activeDim: number,
  filterState: Record<string, Record<number, number> | undefined>,
): CardRow[] {
  const dim = DIMS[activeDim]
  const fs = filterState[dim.key] || {}
  let result = [...cards]
  const tbl = OP_TABLE as Record<string, Record<string, { dtype: string }[]>>
  const inferTbl = INFER_TABLE as Record<string, InferTablePack | undefined>
  const trainTbl = TRAIN_TABLE as Record<string, { framework: string }[] | undefined>
  const commTbl = COMM_TABLE as Record<string, { commType: string }[] | undefined>
  const bwTbl = BW_TABLE as Record<string, BwDetailRow[] | undefined>

  dim.filters.forEach((f, fi) => {
    const ai = fs[fi] ?? 0
    if (ai === 0) return
    const pill = f.pills[ai]
    if (fi === 1 && dim.key === 'op') {
      result = result.filter((c) => {
        const platOps = tbl[c.key] || {}
        return Object.values(platOps).some((rows) => rows.some((r) => r.dtype === pill))
      })
    } else if (fi === 0 && dim.key === 'op') {
      if (pill !== '全部') {
        result = result.filter((c) => {
          const platOps = tbl[c.key] || {}
          return platOps[pill] && platOps[pill].length > 0
        })
      }
    } else if (fi === 0 && dim.key === 'infer') {
      if (pill !== '全部') {
        const inPill = dim.filters[1]?.pills[fs[1] ?? 0]
        result = result.filter((c) => inferPlatHasFilteredRow(c.key, inferTbl, pill, inPill))
      }
    } else if (fi === 1 && dim.key === 'infer') {
      if (pill !== '全部') {
        const batchPill = dim.filters[0]?.pills[fs[0] ?? 0]
        result = result.filter((c) => inferPlatHasFilteredRow(c.key, inferTbl, batchPill, pill))
      }
    } else if (fi === 0 && dim.key === 'train') {
      if (pill !== '全部') {
        result = result.filter((c) => trainPlatHasFramework(c.key, trainTbl, pill))
      }
    } else if (fi === 0 && dim.key === 'comm') {
      if (pill !== '全部') {
        result = result.filter((c) => commPlatHasCommType(c.key, commTbl, pill))
      }
    } else if (fi === 0 && dim.key === 'bw') {
      if (pill !== '全部') {
        result = result.filter((c) => bwPlatHasMode(c.key, bwTbl, pill))
      }
    }
  })
  return result
}

export function parseLatencyMs(v: string | null | undefined): number | null {
  if (!v) return null
  const m = String(v).match(/([0-9.]+)\s*ms/i)
  return m ? Number(m[1]) : null
}

export function avgOpScore(
  rows: { ic?: number; pt?: number; remarks?: string; scoreEligible?: boolean }[],
): number | null {
  const v = rows.filter((r) =>
    canComputeOpRowScore(r.ic ?? NaN, r.pt ?? NaN, r.remarks ?? ''),
  )
  if (!v.length) return null
  return Math.round(
    v.reduce(
      (a, r) => a + (computeOpRowScore(r.ic ?? NaN, r.pt ?? NaN, r.remarks ?? '') ?? 0),
      0,
    ) / v.length,
  )
}
