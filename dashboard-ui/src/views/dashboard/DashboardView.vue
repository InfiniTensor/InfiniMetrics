<script setup lang="ts">
import DashboardCard from '@/components/DashboardCard.vue'
import DashboardComparePanel from '@/views/dashboard/components/DashboardComparePanel.vue'
import DashboardDetailPanel from '@/views/dashboard/components/DashboardDetailPanel.vue'
import DashboardFilterBar from '@/views/dashboard/components/DashboardFilterBar.vue'
import DashboardHeader from '@/views/dashboard/components/DashboardHeader.vue'
import DashboardOverviewCards from '@/views/dashboard/components/DashboardOverviewCards.vue'
import DashboardSidebar from '@/views/dashboard/components/DashboardSidebar.vue'
import { useInfiniDashboard } from '@/composables/useInfiniDashboard'

const { currentView, bcBrand, bcDim, goBack } = useInfiniDashboard()

function onBcOverview() {
  goBack()
}
</script>

<template>
  <!-- 单页三视图：概览 / 详情 / 对比（与 dashboard_preview.html 结构一致） -->
  <div class="container">
    <DashboardHeader />
    <div class="overview-wrap">
      <div class="main-grid">
        <DashboardSidebar />
        <div class="right-panel">
          <!-- 面包屑置于灰底上，与筛选卡片区分开（逻辑同原 DashboardFilterBar 内 global-bc） -->
          <div class="global-bc visible">
            <template v-if="currentView === 'overview'">
              <span class="bc-item cur">概览</span>
            </template>
            <template v-else-if="currentView === 'detail'">
              <span class="bc-item" @click="onBcOverview">概览</span>
              <span class="bc-sep">/</span>
              <span class="bc-item">{{ bcBrand }}</span>
              <span class="bc-sep">/</span>
              <span class="bc-item cur">{{ bcDim }}</span>
            </template>
            <template v-else-if="currentView === 'compare'">
              <span class="bc-item" @click="onBcOverview">概览</span>
              <span class="bc-sep">/</span>
              <span class="bc-item cur">平台对比</span>
            </template>
          </div>
          <DashboardCard class="right-panel-card-filter">
            <DashboardFilterBar />
          </DashboardCard>
          <DashboardCard class="right-panel-card-main">
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
  gap: 16px;
  min-width: 0;
}

.right-panel :deep(.dashboard-card.right-panel-card-filter) {
  flex: 0 0 auto;
  height: auto;
}

.right-panel :deep(.dashboard-card.right-panel-card-main) {
  flex: 1;
  min-height: 0;
}
</style>
