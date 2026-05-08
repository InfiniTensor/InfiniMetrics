<script setup lang="ts">
import { provide } from 'vue'
import { createInfiniDashboardStore } from '@/features/dashboard/createInfiniDashboardStore'
import { INFINI_DASHBOARD_KEY } from '@/features/dashboard/injection'
import DashboardView from '@/views/dashboard/DashboardView.vue'

const dashboardStore = createInfiniDashboardStore()
provide(INFINI_DASHBOARD_KEY, dashboardStore)
</script>

<template>
  <div id="app" class="ib-dashboard-root">
    <DashboardView />
  </div>
</template>
