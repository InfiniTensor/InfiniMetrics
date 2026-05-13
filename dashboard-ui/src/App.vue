<script setup lang="ts">
import { provide } from 'vue'
import { RouterView } from 'vue-router'
import { createInfiniDashboardStore } from '@/features/dashboard/createInfiniDashboardStore'
import { INFINI_DASHBOARD_KEY } from '@/features/dashboard/injection'

const dashboardStore = createInfiniDashboardStore()
provide(INFINI_DASHBOARD_KEY, dashboardStore)

/** 与 `style.css` 中 `--color-primary` 一致，供 ant-design-vue Design Token 使用 */
const antdTheme = {
  token: {
    colorPrimary: '#162b75',
  },
}
</script>

<template>
  <a-config-provider :theme="antdTheme">
    <div id="app" class="ib-dashboard-root">
      <RouterView />
    </div>
  </a-config-provider>
</template>
