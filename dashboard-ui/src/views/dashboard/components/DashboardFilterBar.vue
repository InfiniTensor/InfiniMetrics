<script setup lang="ts">
import { computed } from 'vue'
import { useInfiniDashboard } from '@/composables/useInfiniDashboard'

const store = useInfiniDashboard()
const {
  DIMS,
  activeDim,
  filterState,
  sortDesc,
  comparePlatKeys,
  PLATFORMS,
  currentView,
  bcBrand,
  bcDim,
  toggleSort,
  setFilter,
  clearCompare,
  openComparePage,
  toggleCompare,
  goBack,
} = store

const dim = computed(() => DIMS[activeDim.value])
const fs = computed(() => filterState.value[dim.value.key] || {})

function pillActive(fi: number, pi: number) {
  return (fs.value[fi] ?? 0) === pi
}

const compareChosen = computed(() =>
  comparePlatKeys.value
    .map((k) => PLATFORMS.find((p) => p.key === k)!)
    .filter(Boolean),
)

function onBcOverview() {
  goBack()
}
</script>

<template>
  <div class="filter-bar">
    <!-- 全局面包屑（与 HTML updateGlobalBc 一致） -->
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

    <div
      v-for="(f, fi) in dim.filters"
      :key="f.label"
      class="filter-row"
    >
      <span class="filter-label">{{ f.label }}</span>
      <div class="filter-pills">
        <button
          v-for="(p, pi) in f.pills"
          :key="`${fi}-${pi}-${p}`"
          type="button"
          class="fpill"
          :class="{ active: pillActive(fi, pi) }"
          @click="setFilter(fi, pi)"
        >
          {{ p }}
        </button>
      </div>
      <button
        v-if="fi === 0"
        type="button"
        class="sort-btn"
        @click="toggleSort"
      >
        ⇅ 按得分 {{ sortDesc ? '↓' : '↑' }}
      </button>
    </div>

    <!-- 对比条：有选中平台时展示 -->
    <div class="compare-bar" :class="{ active: compareChosen.length > 0 }">
      <div class="compare-title">已选对比</div>
      <div class="compare-chips">
        <template v-if="compareChosen.length">
          <div v-for="p in compareChosen" :key="p.key" class="compare-chip">
            <span class="compare-dot" :style="{ background: p.color }" />
            {{ p.name }}
            <button type="button" class="compare-remove" @click="toggleCompare(p.key)">×</button>
          </div>
        </template>
        <span v-else class="compare-empty">从卡片中加入 2–4 个平台进行横向对比</span>
      </div>
      <div class="compare-bar-actions">
        <button type="button" class="mini-btn" @click="clearCompare">清空</button>
        <button type="button" class="mini-btn primary" @click="openComparePage">开始对比</button>
      </div>
    </div>
  </div>
</template>
