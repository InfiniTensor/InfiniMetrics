<script setup lang="ts">
import { provide } from 'vue'
import { RouterView } from 'vue-router'
import { createInfiniDashboardStore } from '@/features/dashboard/createInfiniDashboardStore'
import { INFINI_DASHBOARD_KEY } from '@/features/dashboard/injection'

const dashboardStore = createInfiniDashboardStore()
provide(INFINI_DASHBOARD_KEY, dashboardStore)
</script>

<template>
  <div id="app" class="ib-dashboard-root">
    <RouterView />
  </div>
</template>
