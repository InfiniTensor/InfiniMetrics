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
   * 侧栏或顶栏改动平台筛选 / 维度筛选 / 已选对比后，与路由同步。
   * - 对比页：顶栏筛选项仅改 store，**不跳转**（与详情顶栏一致地留在当前视图）。
   * - 对比页 + `fromPlatformNav`：侧栏切换「平台筛选」时回到概览。
   * - 详情页：`replace` 到概览（侧栏切换平台等需退出详情 URL）。
   * - 概览页：仅 `switchMainView('overview')`，避免对 `/` 重复 push。
   */
  function leaveCompareOrSyncOverview(opts?: { fromPlatformNav?: boolean }) {
    const name = router.currentRoute.value.name
    if (name === 'compare') {
      if (opts?.fromPlatformNav) {
        void router.replace({ name: 'overview' })
      }
      return
    }
    if (name === 'detail') {
      void router.replace({ name: 'overview' })
      return
    }
    if (name === 'overview') {
      store.switchMainView('overview')
    }
  }

  return { goOverview, goDetail, goCompare, leaveCompareOrSyncOverview }
}
