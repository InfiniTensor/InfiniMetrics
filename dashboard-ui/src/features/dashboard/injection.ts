import type { InjectionKey } from 'vue'
import type { InfiniDashboardStore } from './createInfiniDashboardStore'

export const INFINI_DASHBOARD_KEY: InjectionKey<InfiniDashboardStore> = Symbol('infiniDashboard')
