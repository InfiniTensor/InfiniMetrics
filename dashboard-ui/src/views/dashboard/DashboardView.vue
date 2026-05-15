<script setup lang="ts">
import { computed, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import DashboardCard from '@/components/DashboardCard.vue'
import DashboardComparePanel from '@/views/dashboard/components/DashboardComparePanel.vue'
import DashboardDetailPanel from '@/views/dashboard/components/DashboardDetailPanel.vue'
import DashboardFilterBar from '@/views/dashboard/components/DashboardFilterBar.vue'
import DashboardHeader from '@/views/dashboard/components/DashboardHeader.vue'
import DashboardOverviewCards from '@/views/dashboard/components/DashboardOverviewCards.vue'
import DashboardSidebar from '@/views/dashboard/components/DashboardSidebar.vue'
import { useInfiniDashboard } from '@/composables/useInfiniDashboard'
import { useDashboardNavigation } from '@/composables/useDashboardNavigation'
import { applyDashboardRoute } from '@/features/dashboard/applyDashboardRoute'
import { DIMS, PLATFORMS } from '@/data'
import { routeParamString } from '@/utils/routeParams'

const route = useRoute()
const router = useRouter()
const store = useInfiniDashboard()
const { currentView, bcBrand, bcDim } = store
const { goOverview } = useDashboardNavigation()

/** 详情页面包屑品牌与 URL 一致（与详情标题同源） */
const bcBrandDisplay = computed(() => {
  if (route.name !== 'detail') return bcBrand.value
  const pk = routeParamString(route.params.platKey)
  return PLATFORMS.find((p) => p.key === pk)?.name ?? bcBrand.value
})

const bcDimDisplay = computed(() => {
  if (route.name !== 'detail') return bcDim.value
  const dk = routeParamString(route.params.dimKey)
  return DIMS.find((d) => d.key === dk)?.label ?? bcDim.value
})

watch(
  () => route.fullPath,
  () => {
    if (route.name === 'detail') {
      const platKey = routeParamString(route.params.platKey)
      const dimKey = routeParamString(route.params.dimKey)
      const platOk = PLATFORMS.some((p) => p.key === platKey)
      const dimOk = DIMS.some((d) => d.key === dimKey)
      if (!platOk || !dimOk) {
        void router.replace({ name: 'overview' })
        return
      }
    }
    applyDashboardRoute(route, store)
  },
  { immediate: true },
)

function onBcOverview() {
  goOverview()
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
              <span class="bc-item">{{ bcBrandDisplay }}</span>
              <span class="bc-sep">/</span>
              <span class="bc-item cur">{{ bcDimDisplay }}</span>
            </template>
            <template v-else-if="currentView === 'compare'">
              <span class="bc-item" @click="onBcOverview">概览</span>
              <span class="bc-sep">/</span>
              <span class="bc-item cur">平台对比</span>
            </template>
          </div>
          <div class="right-panel-filter-shell">
            <DashboardFilterBar />
          </div>
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
  /* 右侧主栏衬底；顶侧不留 padding，避免「概览」面包屑行上方空隙 */
  background: var(--page-bg);
  padding: 0 16px 16px;
  gap: 16px;
  min-width: 0;
}

.right-panel > .global-bc.visible {
  width: 100%;
  box-sizing: border-box;
  padding-top: 0;
  padding-bottom: 10px;
  border-bottom: 1px solid #162b75;
}

.right-panel > .global-bc.visible .bc-item {
  color: #162b75;
  /* 与顶栏 DashboardFilterBar 筛选标签 / pills（fontSizeSM 16）一致 */
  font-size: 16px;
}

.right-panel > .global-bc.visible .bc-item.cur {
  color: #162b75;
  font-weight: 700;
}

.right-panel > .global-bc.visible .bc-sep {
  color: rgba(22, 43, 117, 0.32);
  font-size: 16px;
}

.right-panel > .global-bc.visible .bc-item:hover {
  color: #1a3488;
}

.right-panel .right-panel-filter-shell {
  flex: 0 0 auto;
  height: auto;
  min-width: 0;
  /* 与白卡片主区拉近 3px（不改变面包屑与筛选区间距） */
  margin-bottom: -5px;
}

.right-panel :deep(.dashboard-card.right-panel-card-main) {
  flex: 1;
  min-height: 0;
}
</style>
