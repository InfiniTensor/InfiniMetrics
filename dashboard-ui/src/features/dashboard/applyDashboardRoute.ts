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

      let typeIdx = preservedOpBar?.[0] ?? 0
      let typeLabel = unionPills[Math.min(Math.max(0, typeIdx), unionPills.length - 1)] ?? '全部'
      if (typeIdx === 0 || typeLabel === '全部' || !opKeys.includes(typeLabel)) {
        store.detailState.value.opKey = defaultOpKey
        typeLabel = store.detailState.value.opKey
      } else {
        store.detailState.value.opKey = typeLabel
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

      let precIdx = preservedOpBar?.[1] ?? 0
      let precLabel = precPills[Math.min(Math.max(0, precIdx), precPills.length - 1)] ?? '全部'
      if (precIdx === 0 || precLabel === '全部' || !platDtypes.has(precLabel)) {
        const pickD = precPills.find((lab, i) => i > 0 && platDtypes.has(lab))
        if (pickD) {
          precLabel = pickD
          precIdx = precPills.indexOf(pickD)
        } else {
          const sorted = [...platDtypes].sort((a, b) => a.localeCompare(b))
          const fallback = sorted.find((d) => precPills.includes(d))
          if (fallback) {
            precLabel = fallback
            precIdx = precPills.indexOf(fallback)
          } else if (precPills.length > 1) {
            precIdx = 1
            precLabel = precPills[precIdx]!
          }
        }
      }
      store.detailState.value.prec = precLabel
      store.filterState.value.op = { 0: typeUnionIdx, 1: precIdx }
    }

    const inferTbl = INFER_TABLE as Record<string, InferTablePack | undefined>
    if (dimKey === 'infer') {
      const inferDim = DIMS.find((d) => d.key === 'infer')!
      const batchPills = inferDim.filters[0]!.pills
      const inLenPills = inferDim.filters[1]!.pills
      const batches = inferNumericSetForPlatform(inferTbl, plat.key, 'batch')
      const lens = inferNumericSetForPlatform(inferTbl, plat.key, 'inLen')
      let foundBi = -1
      let foundLi = -1
      const tryInferPair = (requireFilteredRow: boolean) => {
        for (let bi = 1; bi < batchPills.length; bi++) {
          const bLab = batchPills[bi]
          if (bLab === '全部' || !batches.has(Number(bLab))) continue
          for (let li = 1; li < inLenPills.length; li++) {
            const lLab = inLenPills[li]
            if (lLab === '全部' || !lens.has(Number(lLab))) continue
            if (!requireFilteredRow || inferPlatHasFilteredRow(plat.key, inferTbl, bLab, lLab)) {
              foundBi = bi
              foundLi = li
              return
            }
          }
        }
      }
      const pin = preservedInferBar
      if (pin != null) {
        const bi = Math.min(Math.max(0, pin[0] ?? 0), batchPills.length - 1)
        const li = Math.min(Math.max(0, pin[1] ?? 0), inLenPills.length - 1)
        const bLab = batchPills[bi]
        const lLab = inLenPills[li]
        /**
         * 详情顶栏推理筛与算子/通信一致：不展示「全部」pill（见 DashboardFilterBar）。
         * 若沿用概览的索引 0 或「全部」文案，store 里仍是 unionIndex 0，详情区无对应按钮 → 表现为未选中。
         * 仅当 Batch、In-len 均为具体项且在平台数据中存在时才沿用概览选择。
         */
        const batchOk = bi > 0 && bLab !== '全部' && batches.has(Number(bLab))
        const inLenOk = li > 0 && lLab !== '全部' && lens.has(Number(lLab))
        if (batchOk && inLenOk && inferPlatHasFilteredRow(plat.key, inferTbl, bLab, lLab)) {
          foundBi = bi
          foundLi = li
        }
      }
      if (foundBi < 0) {
        tryInferPair(true)
        if (foundBi < 0) tryInferPair(false)
      }
      if (foundBi >= 0 && foundLi >= 0) {
        store.filterState.value.infer = { 0: foundBi, 1: foundLi }
      }
    }

    if (dimKey === 'train') {
      const trainDim = DIMS.find((d) => d.key === 'train')!
      const fwPills = trainDim.filters[0]!.pills
      const trainTbl = TRAIN_TABLE as Record<string, { framework: string }[] | undefined>
      let pickedFw = -1
      const pti = preservedTrainBar?.[0]
      if (pti != null) {
        const idx = Math.min(Math.max(0, pti), fwPills.length - 1)
        const lab = fwPills[idx]
        if (idx === 0 || lab === '全部' || trainPlatHasFramework(plat.key, trainTbl, lab)) {
          pickedFw = idx
        }
      }
      if (pickedFw < 0) {
        for (let i = 1; i < fwPills.length; i++) {
          if (trainPlatHasFramework(plat.key, trainTbl, fwPills[i])) {
            pickedFw = i
            break
          }
        }
      }
      if (pickedFw >= 0) {
        store.filterState.value.train = { 0: pickedFw }
      }
    }

    if (dimKey === 'comm') {
      const commDim = DIMS.find((d) => d.key === 'comm')!
      const typePills = commDim.filters[0]!.pills
      const commTbl = COMM_TABLE as Record<string, { commType: string }[] | undefined>
      let pickedComm = -1
      const pci = preservedCommBar?.[0]
      if (pci != null) {
        const idx = Math.min(Math.max(0, pci), typePills.length - 1)
        const lab = typePills[idx]
        // 详情顶栏与其他维度一致不展示「全部」：概览为「全部」时勿沿用索引 0，交给下方回退逻辑
        if (idx > 0 && lab !== '全部' && commPlatHasCommType(plat.key, commTbl, lab)) {
          pickedComm = idx
        }
      }
      if (pickedComm < 0) {
        for (let i = 1; i < typePills.length; i++) {
          if (commPlatHasCommType(plat.key, commTbl, typePills[i])) {
            pickedComm = i
            break
          }
        }
      }
      if (pickedComm >= 0) {
        store.filterState.value.comm = { 0: pickedComm }
      }
    }

    if (dimKey === 'bw') {
      const bwDim = DIMS.find((d) => d.key === 'bw')!
      const modePills = bwDim.filters[0]!.pills
      const bwTbl = BW_TABLE as Parameters<typeof bwPlatHasMode>[1]
      let pickedBw = -1
      const pbi = preservedBwBar?.[0]
      if (pbi != null) {
        const idx = Math.min(Math.max(0, pbi), modePills.length - 1)
        const lab = modePills[idx]
        if (idx === 0 || lab === '全部' || bwPlatHasMode(plat.key, bwTbl, lab)) {
          pickedBw = idx
        }
      }
      if (pickedBw < 0) {
        for (let i = 1; i < modePills.length; i++) {
          if (bwPlatHasMode(plat.key, bwTbl, modePills[i])) {
            pickedBw = i
            break
          }
        }
      }
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
