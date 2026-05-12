<script setup lang="ts">
import { Modal } from 'ant-design-vue'
import { computed } from 'vue'
import { useInfiniDashboard } from '@/composables/useInfiniDashboard'
import { useDashboardNavigation } from '@/composables/useDashboardNavigation'
import { OP_TABLE, INFER_TABLE, TRAIN_TABLE, dtypesForOpPlatform } from '@/data'
import { inferNumericSetForPlatform, type InferTablePack } from '@/features/dashboard/inferBenchmark'

const store = useInfiniDashboard()
const {
  DIMS,
  activeDim,
  filterState,
  currentView,
  detailState,
  sortDesc,
  comparePlatKeys,
  PLATFORMS,
  toggleSort,
  setFilter,
  clearCompare,
  compareCards,
  toggleCompare,
} = store
const { goCompare, leaveCompareOrSyncOverview } = useDashboardNavigation()

function onSetFilter(fi: number, pi: number) {
  setFilter(fi, pi)
  leaveCompareOrSyncOverview()
}

function onToggleCompareClose(platKey: string) {
  toggleCompare(platKey)
  leaveCompareOrSyncOverview()
}

function onClearCompare() {
  clearCompare()
  leaveCompareOrSyncOverview()
}

function onOpenComparePage() {
  if (compareCards.value.length < 2) {
    Modal.warning({ title: '提示', content: '请至少选择 2 个有当前维度数据的平台' })
    return
  }
  void goCompare()
}

const dim = computed(() => DIMS[activeDim.value])
const fs = computed(() => filterState.value[dim.value.key] || {})

const inferTableForBar = INFER_TABLE as Record<string, InferTablePack | undefined>
const trainTableForBar = TRAIN_TABLE as Record<string, { framework: string }[] | undefined>

/** 顶栏 pills：`setFilter` 始终传并集维度下的索引；算子 / 推理详情仅展示当前芯片在并集中的子集 */
const filterBarFilterRows = computed(() => {
  const d = dim.value
  const view = currentView.value
  const plat = detailState.value.platKey
  const opTbl = OP_TABLE as Record<string, Record<string, unknown>>
  return d.filters.map((f, fi) => {
    const pills = f.pills.map((label, unionIndex) => ({ label, unionIndex }))
    if (d.key === 'op' && fi === 0 && view === 'detail') {
      const platKeys = new Set(Object.keys(opTbl[plat] || {}))
      return {
        label: f.label,
        pills: pills.filter((x) => x.label !== '全部' && platKeys.has(x.label)),
      }
    }
    if (d.key === 'op' && fi === 1 && view === 'detail') {
      const platDtypes = dtypesForOpPlatform(OP_TABLE, plat)
      return {
        label: f.label,
        pills: pills.filter((x) => x.label !== '全部' && platDtypes.has(x.label)),
      }
    }
    if (d.key === 'infer' && fi === 0 && view === 'detail') {
      const batches = inferNumericSetForPlatform(inferTableForBar, plat, 'batch')
      return {
        label: f.label,
        pills: pills.filter((x) => x.label === '全部' || batches.has(Number(x.label))),
      }
    }
    if (d.key === 'infer' && fi === 1 && view === 'detail') {
      const lens = inferNumericSetForPlatform(inferTableForBar, plat, 'inLen')
      return {
        label: f.label,
        pills: pills.filter((x) => x.label === '全部' || lens.has(Number(x.label))),
      }
    }
    if (d.key === 'train' && fi === 0 && view === 'detail') {
      const fwOnPlat = new Set(
        (trainTableForBar[plat] || [])
          .map((r) => String(r.framework || '').trim().toLowerCase())
          .filter(Boolean),
      )
      return {
        label: f.label,
        pills: pills.filter((x) => x.label === '全部' || fwOnPlat.has(x.label.toLowerCase())),
      }
    }
    return { label: f.label, pills }
  })
})

const compareChosen = computed(() =>
  comparePlatKeys.value
    .map((k) => PLATFORMS.find((p) => p.key === k)!)
    .filter(Boolean),
)

/** 筛选区（含已选对比）Ant Design 主题：圆角 2px；fontSizeSM 供 Tag 与 small 按钮对齐 */
const filterBarTheme = {
  token: {
    borderRadius: 2,
    borderRadiusLG: 2,
    borderRadiusSM: 2,
    fontSizeSM: 12,
  },
}

</script>

<template>
  <a-config-provider :theme="filterBarTheme">
  <div
    class="filter-bar-root"
    :style="{ '--filter-bar-font-sm': `${filterBarTheme.token.fontSizeSM}px` }"
  >
    <div
      v-for="(f, fi) in filterBarFilterRows"
      :key="f.label"
      class="filter-row"
    >
      <span class="filter-label">{{ f.label }}</span>
      <div class="filter-pills">
        <a-space :size="[6, 4]" wrap>
          <a-button
            v-for="item in f.pills"
            :key="`${fi}-${item.unionIndex}-${item.label}`"
            size="small"
            :type="(fs[fi] ?? 0) === item.unionIndex ? 'primary' : 'default'"
            @click="onSetFilter(fi, item.unionIndex)"
          >
            {{ item.label }}
          </a-button>
        </a-space>
      </div>
      <div v-if="fi === 0 && currentView === 'overview'" class="filter-row-sort-wrap">
        <a-button size="small" @click="toggleSort">
          ⇅ 按得分 {{ sortDesc ? '↓' : '↑' }}
        </a-button>
      </div>
    </div>

    <!-- 概览 / 对比页展示；详情页不展示（业务上详情无「已选对比」） -->
    <div
      v-if="currentView !== 'detail'"
      class="compare-bar"
      :class="{ active: compareChosen.length > 0 }"
    >
      <div class="compare-title">已选对比</div>
      <div class="compare-chips">
        <template v-if="compareChosen.length">
          <a-tag
            v-for="p in compareChosen"
            :key="p.key"
            closable
            @close="() => onToggleCompareClose(p.key)"
          >
            {{ p.name }}
          </a-tag>
        </template>
        <span v-else class="compare-empty">从卡片中加入 2–4 个平台进行横向对比</span>
      </div>
      <div class="compare-bar-actions">
        <a-space wrap>
          <a-button @click="onClearCompare">清空</a-button>
          <a-button type="primary" @click="onOpenComparePage">开始对比</a-button>
        </a-space>
      </div>
    </div>
  </div>
  </a-config-provider>
</template>

<style scoped>
/* 仅布局：排序按钮靠右，不改变 Ant Design 按钮外观 */
.filter-row-sort-wrap {
  margin-left: auto;
}

/* small 按钮默认仍用 fontSize(14)；Tag 用 fontSizeSM — 筛选行内 small 与 Tag 字号一致 */
.filter-bar-root :deep(.filter-row .ant-btn-sm) {
  font-size: var(--filter-bar-font-sm);
}
</style>
