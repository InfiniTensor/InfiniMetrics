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

  return { goOverview, goDetail, goCompare }
}
