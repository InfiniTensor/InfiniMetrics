import { DIMS, OP_TABLE } from '@/data'

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
      result = result.filter((c) => c.extra && c.extra.includes('batch=' + pill))
    } else if (fi === 1 && dim.key === 'infer') {
      result = result.filter((c) => c.extra && c.extra.includes('in=' + pill))
    }
  })
  return result
}

export function parseLatencyMs(v: string | null | undefined): number | null {
  if (!v) return null
  const m = String(v).match(/([0-9.]+)\s*ms/i)
  return m ? Number(m[1]) : null
}

export function avgOpScore(rows: { ic?: number; pt?: number }[]): number | null {
  const v = rows.filter((r) => r.ic && r.pt)
  if (!v.length) return null
  return Math.round(v.reduce((a, r) => a + (r.pt! / r.ic!) * 100, 0) / v.length)
}
