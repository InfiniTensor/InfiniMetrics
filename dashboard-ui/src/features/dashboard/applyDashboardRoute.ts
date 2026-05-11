import type { RouteLocationNormalizedLoaded } from 'vue-router'
import type { InfiniDashboardStore } from '@/features/dashboard/createInfiniDashboardStore'
import { DIMS, PLATFORMS } from '@/data'
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

    store.activeDim.value = dimIdx
    store.filterState.value = {}
    store.detailState.value.platKey = plat.key
    store.detailState.value.opKey = 'CausalSoftmax'
    store.detailState.value.prec = '全部'
    store.detailState.value.inferTab = 'prefill'
    store.detailTableTab.value = 'data'
    store.bcBrand.value = plat.name
    store.bcDim.value = DIMS[dimIdx].label
    store.switchMainView('detail')
    return
  }

  store.switchMainView('overview')
}
