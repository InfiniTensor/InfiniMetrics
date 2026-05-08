<script setup lang="ts">
import { ClockCircleOutlined } from '@ant-design/icons-vue'
import { DATA_UPDATED_AT } from '@/data'
import { useInfiniDashboard } from '@/composables/useInfiniDashboard'

const {
  PLATFORMS,
  DIMS,
  selectedPlatKeys,
  activeDim,
  togglePlat,
  selectAll,
  selectDomestic,
  setDim,
} = useInfiniDashboard()

function isBrandActive(key: string) {
  return selectedPlatKeys.value.includes(key)
}
</script>

<template>
  <aside class="sidebar">
    <div class="sidebar-scroll">
      <div class="sb-section">
        <div class="sb-heading">平台筛选</div>
        <div class="quick-btns">
          <button type="button" class="q-btn" @click="selectAll">全选</button>
          <button type="button" class="q-btn primary" @click="selectDomestic">只选国产</button>
        </div>
        <div class="brand-list">
          <div
            v-for="p in PLATFORMS"
            :key="p.key"
            class="brand-item"
            :class="{ active: isBrandActive(p.key) }"
            @click="togglePlat(p.key)"
          >
            <div class="brand-logo" :style="{ background: p.color }">{{ p.logo }}</div>
            <div>
              <div class="brand-name">{{ p.name }}</div>
              <div class="brand-type">{{ p.type }}</div>
            </div>
            <div class="brand-check">{{ isBrandActive(p.key) ? '✓' : '' }}</div>
          </div>
        </div>
      </div>
      <div class="sb-section">
        <div class="sb-title">测试维度</div>
        <div class="dim-list">
          <div
            v-for="(d, i) in DIMS"
            :key="d.key"
            class="dim-item"
            :class="{ active: activeDim === i }"
            @click="setDim(i)"
          >
            {{ d.label }}
          </div>
        </div>
      </div>
      <div class="timestamp">
        <ClockCircleOutlined style="font-size: 11px" />
        数据更新于 {{ DATA_UPDATED_AT }}
      </div>
    </div>
  </aside>
</template>
