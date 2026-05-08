import { DIMS, INFER_TABLE, OP_TABLE, TRAIN_TABLE, COMM_TABLE, BW_TABLE } from '@/data'
import { bwPlatHasMode, type BwDetailRow } from '@/features/dashboard/bwBenchmark'
import { commPlatHasCommType } from '@/features/dashboard/commBenchmark'
import { inferPlatHasFilteredRow } from '@/features/dashboard/inferBenchmark'
import { trainPlatHasFramework } from '@/features/dashboard/trainBenchmark'

/** 卡片行（与各维度 CARD_DATA 元素一致的最小字段） */
export type CardRow = {
  key: string
  extra?: string
  ownScore?: number | null
  ownVal?: string | null
  openVal?: string | null
  openScore?: number | null
  n?: number
  ownFw?: string
  openFw?: string
  adv?: boolean
  advTxt?: string
}

/** 与 HTML `applyCardFilter` 行为一致 */
export function applyCardFilter(
  cards: CardRow[],
  activeDim: number,
  filterState: Record<string, Record<number, number> | undefined>,
): CardRow[] {
  const dim = DIMS[activeDim]
  const fs = filterState[dim.key] || {}
  let result = [...cards]
  const tbl = OP_TABLE as Record<string, Record<string, { dtype: string }[]>>
  const inferTbl = INFER_TABLE as Record<
    string,
    { prefill?: { batch: number; inLen: number }[] } | undefined
  >
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
        result = result.filter((c) =>
          inferPlatHasFilteredRow(c.key, inferTbl, pill, inPill),
        )
      }
    } else if (fi === 1 && dim.key === 'infer') {
      if (pill !== '全部') {
        const batchPill = dim.filters[0]?.pills[fs[0] ?? 0]
        result = result.filter((c) =>
          inferPlatHasFilteredRow(c.key, inferTbl, batchPill, pill),
        )
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
  const v = rows.filter((r) => {
    if (r.scoreEligible === false) return false
    return r.ic && r.pt
  })
  if (!v.length) return null
  return Math.round(v.reduce((a, r) => a + (r.pt! / r.ic!) * 100, 0) / v.length)
}
