import { inject } from 'vue'
import { INFINI_DASHBOARD_KEY } from '@/features/dashboard/injection'
import type { InfiniDashboardStore } from '@/features/dashboard/createInfiniDashboardStore'

/** 注入仪表盘 store（须在 App 层 provide） */
export function useInfiniDashboard(): InfiniDashboardStore {
  const s = inject(INFINI_DASHBOARD_KEY)
  if (!s) throw new Error('useInfiniDashboard: 未 provide INFINI_DASHBOARD_KEY')
  return s
}
