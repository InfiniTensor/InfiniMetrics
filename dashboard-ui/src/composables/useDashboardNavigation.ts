import { useRouter } from 'vue-router'
import { DIMS } from '@/data'
import { useInfiniDashboard } from '@/composables/useInfiniDashboard'

/** 与路由绑定的仪表盘导航（详情 URL 可刷新保持） */
export function useDashboardNavigation() {
  const router = useRouter()
  const store = useInfiniDashboard()

  function goOverview() {
    void router.push({ name: 'overview' })
  }

  function goDetail(platKey: string) {
    const dimKey = DIMS[store.activeDim.value].key
    void router.push({
      name: 'detail',
      params: { platKey, dimKey },
    })
  }

  function goCompare() {
    void router.push({ name: 'compare' })
  }

  /**
   * 侧栏或顶栏改动平台筛选 / 维度筛选 / 已选对比后，非「仅概览」路由时一律回到概览。
   * - 对比页、详情页：`replace` 到概览 URL（由路由 watch 同步 store）。
   * - 概览页：仅 `switchMainView('overview')`，避免对 `/` 重复 push 导致导航不触发。
   */
  function leaveCompareOrSyncOverview() {
    const name = router.currentRoute.value.name
    if (name === 'compare' || name === 'detail') {
      void router.replace({ name: 'overview' })
      return
    }
    if (name === 'overview') {
      store.switchMainView('overview')
    }
  }

  return { goOverview, goDetail, goCompare, leaveCompareOrSyncOverview }
}
