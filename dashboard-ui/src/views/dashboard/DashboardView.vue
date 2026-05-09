<script setup lang="ts">
import DashboardCard from '@/components/DashboardCard.vue'
import DashboardComparePanel from '@/views/dashboard/components/DashboardComparePanel.vue'
import DashboardDetailPanel from '@/views/dashboard/components/DashboardDetailPanel.vue'
import DashboardFilterBar from '@/views/dashboard/components/DashboardFilterBar.vue'
import DashboardHeader from '@/views/dashboard/components/DashboardHeader.vue'
import DashboardOverviewCards from '@/views/dashboard/components/DashboardOverviewCards.vue'
import DashboardSidebar from '@/views/dashboard/components/DashboardSidebar.vue'
import { useInfiniDashboard } from '@/composables/useInfiniDashboard'

const { currentView } = useInfiniDashboard()
</script>

<template>
  <!-- 单页三视图：概览 / 详情 / 对比（与 dashboard_preview.html 结构一致） -->
  <div class="container">
    <DashboardHeader />
    <div class="overview-wrap">
      <div class="main-grid">
        <DashboardSidebar />
        <div class="right-panel">
          <DashboardCard>
            <DashboardFilterBar />
            <div class="content-panel">
              <div class="overview-panel" :class="{ active: currentView === 'overview' }">
                <DashboardOverviewCards />
              </div>
              <div class="detail-wrap" :class="{ active: currentView === 'detail' }">
                <DashboardDetailPanel />
              </div>
              <div class="compare-wrap" :class="{ active: currentView === 'compare' }">
                <DashboardComparePanel />
              </div>
            </div>
          </DashboardCard>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.right-panel {
  background: #f7f8fa;
  padding: 20px;
}

.right-panel :deep(.dashboard-card) {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
</style>
