import type { RouteLocationNormalizedLoaded } from 'vue-router'
import type { InfiniDashboardStore } from '@/features/dashboard/createInfiniDashboardStore'
import { DIMS, OP_TABLE, PLATFORMS, dtypesForOpPlatform } from '@/data'
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

    /** 进入详情前概览顶栏的算子筛选（勿在清空 filterState 之后再读） */
    const preservedOpBar = store.filterState.value.op

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
    store.bcBrand.value = plat.name
    store.bcDim.value = DIMS[dimIdx].label
    store.switchMainView('detail')
    return
  }

  store.switchMainView('overview')
}
