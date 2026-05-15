import type { RouteLocationNormalizedLoaded } from 'vue-router'
import type { InfiniDashboardStore } from '@/features/dashboard/createInfiniDashboardStore'
import { DIMS, OP_TABLE, PLATFORMS, dtypesForOpPlatform, INFER_TABLE, TRAIN_TABLE, COMM_TABLE, BW_TABLE } from '@/data'
import { bwPlatHasMode } from '@/features/dashboard/bwBenchmark'
import { commPlatHasCommType } from '@/features/dashboard/commBenchmark'
import {
  inferNumericSetForPlatform,
  inferPlatHasFilteredRow,
  type InferTablePack,
} from '@/features/dashboard/inferBenchmark'
import { trainPlatHasFramework } from '@/features/dashboard/trainBenchmark'
import { routeParamString } from '@/utils/routeParams'

/** 概览 pill 为具体项（非「全部」、unionIndex > 0） */
function isSpecificOverviewPill(idx: number, label: string | undefined): boolean {
  return idx > 0 && label != null && label !== '全部'
}

/** 详情顶栏不展示「全部」：取概览已选具体项，否则该平台第一个可用项 */
function pickDetailFilterIndex(
  pills: string[],
  preservedIdx: number | undefined,
  isValid: (label: string) => boolean,
): number {
  const idx = Math.min(Math.max(0, preservedIdx ?? 0), pills.length - 1)
  const lab = pills[idx]
  if (isSpecificOverviewPill(idx, lab) && isValid(lab)) {
    return idx
  }
  for (let i = 1; i < pills.length; i++) {
    const p = pills[i]
    if (p !== '全部' && isValid(p)) return i
  }
  return -1
}

function firstInferPairOnPlat(
  platKey: string,
  inferTbl: Record<string, InferTablePack | undefined>,
  batchPills: string[],
  inLenPills: string[],
  batches: Set<number>,
  lens: Set<number>,
  requireRow: boolean,
): { bi: number; li: number } | null {
  for (let bi = 1; bi < batchPills.length; bi++) {
    const bLab = batchPills[bi]
    if (bLab === '全部' || !batches.has(Number(bLab))) continue
    for (let li = 1; li < inLenPills.length; li++) {
      const lLab = inLenPills[li]
      if (lLab === '全部' || !lens.has(Number(lLab))) continue
      if (!requireRow || inferPlatHasFilteredRow(platKey, inferTbl, bLab, lLab)) {
        return { bi, li }
      }
    }
  }
  return null
}

/** 概览推理筛选项带入详情：支持仅 Batch / 仅 In-len / 两者皆非「全部」 */
function resolveInferDetailFilters(
  platKey: string,
  inferTbl: Record<string, InferTablePack | undefined>,
  preserved: Record<number, number> | undefined,
): { 0: number; 1: number } | null {
  const inferDim = DIMS.find((d) => d.key === 'infer')!
  const batchPills = inferDim.filters[0]!.pills
  const inLenPills = inferDim.filters[1]!.pills
  const batches = inferNumericSetForPlatform(inferTbl, platKey, 'batch')
  const lens = inferNumericSetForPlatform(inferTbl, platKey, 'inLen')

  const bi = Math.min(Math.max(0, preserved?.[0] ?? 0), batchPills.length - 1)
  const li = Math.min(Math.max(0, preserved?.[1] ?? 0), inLenPills.length - 1)
  const bLab = batchPills[bi]
  const lLab = inLenPills[li]
  const batchSpecific = isSpecificOverviewPill(bi, bLab) && batches.has(Number(bLab))
  const inLenSpecific = isSpecificOverviewPill(li, lLab) && lens.has(Number(lLab))

  if (batchSpecific && inLenSpecific && inferPlatHasFilteredRow(platKey, inferTbl, bLab, lLab)) {
    return { 0: bi, 1: li }
  }
  if (batchSpecific) {
    for (let lj = 1; lj < inLenPills.length; lj++) {
      const l = inLenPills[lj]
      if (l === '全部' || !lens.has(Number(l))) continue
      if (inferPlatHasFilteredRow(platKey, inferTbl, bLab, l)) {
        return { 0: bi, 1: lj }
      }
    }
  }
  if (inLenSpecific) {
    for (let bj = 1; bj < batchPills.length; bj++) {
      const b = batchPills[bj]
      if (b === '全部' || !batches.has(Number(b))) continue
      if (inferPlatHasFilteredRow(platKey, inferTbl, b, lLab)) {
        return { 0: bj, 1: li }
      }
    }
  }

  const pair =
    firstInferPairOnPlat(platKey, inferTbl, batchPills, inLenPills, batches, lens, true) ??
    firstInferPairOnPlat(platKey, inferTbl, batchPills, inLenPills, batches, lens, false)
  return pair ? { 0: pair.bi, 1: pair.li } : null
}

/** 根据当前 URL 同步仪表盘视图与详情上下文（刷新 / 直达链接 / 浏览器前进后退） */
export function applyDashboardRoute(
  route: RouteLocationNormalizedLoaded,
  store: InfiniDashboardStore,
): void {
  if (route.name === 'compare') {
    store.switchMainView('compare')
    return
  }

  if (route.name === 'detail') {
    const platKey = routeParamString(route.params.platKey)
    const dimKey = routeParamString(route.params.dimKey)
    const plat = PLATFORMS.find((p) => p.key === platKey)
    const dimIdx = DIMS.findIndex((d) => d.key === dimKey)
    if (!plat || dimIdx < 0) {
      return
    }

    /** 进入详情前概览顶栏筛选（勿在清空 filterState 之后再读） */
    const preservedOpBar = store.filterState.value.op
    const preservedInferBar = store.filterState.value.infer
    const preservedTrainBar = store.filterState.value.train
    const preservedCommBar = store.filterState.value.comm
    const preservedBwBar = store.filterState.value.bw

    const prevDimIdx = store.activeDim.value
    if (dimIdx !== prevDimIdx) {
      store.resetCompareToDefault()
    }
    store.activeDim.value = dimIdx
    store.filterState.value = {}
    store.detailState.value.platKey = plat.key
    const platOps = OP_TABLE as Record<string, Record<string, unknown>>
    const opKeys = Object.keys(platOps[plat.key] || {})
    const defaultOpKey = opKeys[0] ?? 'CausalSoftmax'
    store.detailState.value.opKey = defaultOpKey
    store.detailState.value.prec = '全部'
    store.detailState.value.inferTab = 'prefill'

    if (dimKey === 'op') {
      const opDim = DIMS.find((d) => d.key === 'op')!
      const unionPills = opDim.filters[0]!.pills
      const precPills = opDim.filters[1]!.pills
      const platDtypes = dtypesForOpPlatform(OP_TABLE, plat.key)

      const typeIdxRaw = preservedOpBar?.[0] ?? 0
      const typeLabel =
        unionPills[Math.min(Math.max(0, typeIdxRaw), unionPills.length - 1)] ?? '全部'
      if (isSpecificOverviewPill(typeIdxRaw, typeLabel) && opKeys.includes(typeLabel)) {
        store.detailState.value.opKey = typeLabel
      } else {
        store.detailState.value.opKey = defaultOpKey
      }
      let typeUnionIdx = unionPills.indexOf(store.detailState.value.opKey)
      if (typeUnionIdx <= 0) {
        const pick = unionPills.find((lab, i) => i > 0 && opKeys.includes(lab))
        if (pick) {
          store.detailState.value.opKey = pick
          typeUnionIdx = unionPills.indexOf(pick)
        } else {
          typeUnionIdx = Math.min(1, unionPills.length - 1)
        }
      }

      const precIdx = pickDetailFilterIndex(
        precPills,
        preservedOpBar?.[1],
        (lab) => platDtypes.has(lab),
      )
      if (precIdx >= 0) {
        store.detailState.value.prec = precPills[precIdx]!
        store.filterState.value.op = { 0: typeUnionIdx, 1: precIdx }
      } else {
        store.filterState.value.op = { 0: typeUnionIdx, 1: 0 }
      }
    }

    const inferTbl = INFER_TABLE as Record<string, InferTablePack | undefined>
    if (dimKey === 'infer') {
      const inferFilters = resolveInferDetailFilters(plat.key, inferTbl, preservedInferBar)
      if (inferFilters) {
        store.filterState.value.infer = inferFilters
      }
    }

    if (dimKey === 'train') {
      const trainDim = DIMS.find((d) => d.key === 'train')!
      const fwPills = trainDim.filters[0]!.pills
      const trainTbl = TRAIN_TABLE as Record<string, { framework: string }[] | undefined>
      const pickedFw = pickDetailFilterIndex(fwPills, preservedTrainBar?.[0], (lab) =>
        trainPlatHasFramework(plat.key, trainTbl, lab),
      )
      if (pickedFw >= 0) {
        store.filterState.value.train = { 0: pickedFw }
      }
    }

    if (dimKey === 'comm') {
      const commDim = DIMS.find((d) => d.key === 'comm')!
      const typePills = commDim.filters[0]!.pills
      const commTbl = COMM_TABLE as Record<string, { commType: string }[] | undefined>
      const pickedComm = pickDetailFilterIndex(typePills, preservedCommBar?.[0], (lab) =>
        commPlatHasCommType(plat.key, commTbl, lab),
      )
      if (pickedComm >= 0) {
        store.filterState.value.comm = { 0: pickedComm }
      }
    }

    if (dimKey === 'bw') {
      const bwDim = DIMS.find((d) => d.key === 'bw')!
      const modePills = bwDim.filters[0]!.pills
      const bwTbl = BW_TABLE as Parameters<typeof bwPlatHasMode>[1]
      const pickedBw = pickDetailFilterIndex(modePills, preservedBwBar?.[0], (lab) =>
        bwPlatHasMode(plat.key, bwTbl, lab),
      )
      if (pickedBw >= 0) {
        store.filterState.value.bw = { 0: pickedBw }
      }
    }

    store.bcBrand.value = plat.name
    store.bcDim.value = DIMS[dimIdx].label
    store.switchMainView('detail')
    return
  }

  store.switchMainView('overview')
}
