<script setup lang="ts">
import { Modal } from 'ant-design-vue'
import { computed } from 'vue'
import { useInfiniDashboard } from '@/composables/useInfiniDashboard'
import { useDashboardNavigation } from '@/composables/useDashboardNavigation'

const store = useInfiniDashboard()
const {
  DIMS,
  activeDim,
  filterState,
  sortDesc,
  comparePlatKeys,
  PLATFORMS,
  toggleSort,
  setFilter,
  clearCompare,
  compareCards,
  toggleCompare,
} = store
const { goCompare } = useDashboardNavigation()

function onOpenComparePage() {
  if (compareCards.value.length < 2) {
    Modal.warning({ title: '提示', content: '请至少选择 2 个有当前维度数据的平台' })
    return
  }
  void goCompare()
}

const dim = computed(() => DIMS[activeDim.value])
const fs = computed(() => filterState.value[dim.value.key] || {})

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
      v-for="(f, fi) in dim.filters"
      :key="f.label"
      class="filter-row"
    >
      <span class="filter-label">{{ f.label }}</span>
      <div class="filter-pills">
        <a-space :size="[6, 4]" wrap>
          <a-button
            v-for="(p, pi) in f.pills"
            :key="`${fi}-${pi}-${p}`"
            size="small"
            :type="(fs[fi] ?? 0) === pi ? 'primary' : 'default'"
            @click="setFilter(fi, pi)"
          >
            {{ p }}
          </a-button>
        </a-space>
      </div>
      <div v-if="fi === 0" class="filter-row-sort-wrap">
        <a-button size="small" @click="toggleSort">
          ⇅ 按得分 {{ sortDesc ? '↓' : '↑' }}
        </a-button>
      </div>
    </div>

    <!-- 对比条：与算子类型 / 精度等同在一卡片内 -->
    <div class="compare-bar" :class="{ active: compareChosen.length > 0 }">
      <div class="compare-title">已选对比</div>
      <div class="compare-chips">
        <template v-if="compareChosen.length">
          <a-tag
            v-for="p in compareChosen"
            :key="p.key"
            closable
            @close="() => toggleCompare(p.key)"
          >
            {{ p.name }}
          </a-tag>
        </template>
        <span v-else class="compare-empty">从卡片中加入 2–4 个平台进行横向对比</span>
      </div>
      <div class="compare-bar-actions">
        <a-space wrap>
          <a-button @click="clearCompare">清空</a-button>
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
