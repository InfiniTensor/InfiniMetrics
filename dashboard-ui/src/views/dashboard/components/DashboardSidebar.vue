<script setup lang="ts">
import { computed } from 'vue'
import { BENCHMARK_DATA_META, DATA_UPDATED_AT, INFER_TABLE, OP_TABLE, TRAIN_TABLE, COMM_TABLE, BW_TABLE } from '@/data'
import {
  formatOperatorCsvDateDisplay,
  maxOperatorCsvDateForPlatform,
  type OperatorTableRow,
} from '@/features/dashboard/operatorBenchmark'
import { maxInferCsvDateForPlatform, type InferTablePack } from '@/features/dashboard/inferBenchmark'
import { maxTrainCsvDateForPlatform } from '@/features/dashboard/trainBenchmark'
import { maxCommCsvDateForPlatform } from '@/features/dashboard/commBenchmark'
import { maxBwCsvDateForPlatform } from '@/features/dashboard/bwBenchmark'
import { useInfiniDashboard } from '@/composables/useInfiniDashboard'
import { useDashboardNavigation } from '@/composables/useDashboardNavigation'
import { platIconSrc } from '@/features/dashboard/platPublicIcon'

const {
  PLATFORMS,
  DIMS,
  selectedPlatKeys,
  activeDim,
  currentView,
  detailState,
  togglePlat,
  selectAll,
  selectDomestic,
  setDim,
} = useInfiniDashboard()
const { goOverview, leaveCompareOrSyncOverview } = useDashboardNavigation()

/** 与 store 中「全选」语义一致：当前已包含全部平台 */
const quickSelectAllActive = computed(() => {
  const sel = selectedPlatKeys.value
  if (sel.length !== PLATFORMS.length) return false
  return PLATFORMS.every((p) => sel.includes(p.key))
})

/** 与 store 中「国产」语义一致：选中集合恰为全部国产平台 */
const quickDomesticActive = computed(() => {
  const domestic = new Set(PLATFORMS.filter((p) => p.domestic).map((p) => p.key))
  const sel = selectedPlatKeys.value
  if (sel.length !== domestic.size) return false
  return sel.every((k) => domestic.has(k))
})

function onSetDim(i: number) {
  setDim(i)
  void goOverview()
}

function onTogglePlat(key: string) {
  togglePlat(key)
  leaveCompareOrSyncOverview()
}

function onSelectAll() {
  selectAll()
  leaveCompareOrSyncOverview()
}

function onSelectDomestic() {
  selectDomestic()
  leaveCompareOrSyncOverview()
}

/** 标题下小字：国际标杆 / 国产（与 PLATFORMS.type 一致，缺省时按 domestic 推断） */
function platTagline(p: (typeof PLATFORMS)[number]) {
  return p.type || (p.domestic ? '国产' : '国际标杆')
}

/** 侧栏主文案与设计稿一致（不改 PLATFORMS 数据，仅展示层） */
function platLineName(p: (typeof PLATFORMS)[number]) {
  if (p.key === 'nvidia') return 'NVIDIA'
  if (p.key === 'generic') return '阿里'
  return p.name
}

/** 侧栏「数据更新于」：算子 / 推理维度走 CSV date 与生成元数据；其余用全局占位 */
const dataUpdatedAtDisplay = computed(() => {
  const dimKey = DIMS[activeDim.value]?.key
  if (dimKey === 'infer') {
    const meta = (BENCHMARK_DATA_META as { inferDatasetUpdatedAt?: string }).inferDatasetUpdatedAt
    if (currentView.value === 'detail') {
      const plat = detailState.value.platKey
      const maxRow = maxInferCsvDateForPlatform(
        INFER_TABLE as Record<string, InferTablePack | undefined>,
        plat,
      )
      if (maxRow) return formatOperatorCsvDateDisplay(maxRow)
    }
    if (meta) return formatOperatorCsvDateDisplay(meta)
    return formatOperatorCsvDateDisplay(DATA_UPDATED_AT)
  }
  if (dimKey === 'train') {
    const meta = (BENCHMARK_DATA_META as { trainDatasetUpdatedAt?: string }).trainDatasetUpdatedAt
    if (currentView.value === 'detail') {
      const plat = detailState.value.platKey
      const maxRow = maxTrainCsvDateForPlatform(
        TRAIN_TABLE as Record<string, { date?: string }[] | undefined>,
        plat,
      )
      if (maxRow) return formatOperatorCsvDateDisplay(maxRow)
    }
    if (meta) return formatOperatorCsvDateDisplay(meta)
    return formatOperatorCsvDateDisplay(DATA_UPDATED_AT)
  }
  if (dimKey === 'comm') {
    const meta = (BENCHMARK_DATA_META as { commDatasetUpdatedAt?: string }).commDatasetUpdatedAt
    if (currentView.value === 'detail') {
      const plat = detailState.value.platKey
      const maxRow = maxCommCsvDateForPlatform(
        COMM_TABLE as Record<string, { date?: string }[] | undefined>,
        plat,
      )
      if (maxRow) return formatOperatorCsvDateDisplay(maxRow)
    }
    if (meta) return formatOperatorCsvDateDisplay(meta)
    return formatOperatorCsvDateDisplay(DATA_UPDATED_AT)
  }
  if (dimKey === 'bw') {
    const meta = (BENCHMARK_DATA_META as { bwDatasetUpdatedAt?: string }).bwDatasetUpdatedAt
    if (currentView.value === 'detail') {
      const plat = detailState.value.platKey
      const maxRow = maxBwCsvDateForPlatform(
        BW_TABLE as Record<string, { date?: string }[] | undefined>,
        plat,
      )
      if (maxRow) return formatOperatorCsvDateDisplay(maxRow)
    }
    if (meta) return formatOperatorCsvDateDisplay(meta)
    return formatOperatorCsvDateDisplay(DATA_UPDATED_AT)
  }
  if (dimKey !== 'op') return formatOperatorCsvDateDisplay(DATA_UPDATED_AT)
  const meta = (BENCHMARK_DATA_META as { opDatasetUpdatedAt?: string }).opDatasetUpdatedAt
  if (currentView.value === 'detail') {
    const plat = detailState.value.platKey
    const tbl = OP_TABLE as unknown as Record<string, Record<string, OperatorTableRow[]>>
    const maxRow = maxOperatorCsvDateForPlatform(tbl[plat])
    if (maxRow) return formatOperatorCsvDateDisplay(maxRow)
  }
  if (meta) return formatOperatorCsvDateDisplay(meta)
  return formatOperatorCsvDateDisplay(DATA_UPDATED_AT)
})
</script>

<template>
  <aside class="sidebar">
    <div class="sidebar-scroll">
      <div class="sidebar-body">
        <div class="sb-section">
          <div class="sb-section-head">
            <span class="sb-section-title">平台筛选</span>
            <div class="sb-section-line" />
          </div>
          <div class="quick-btns">
            <button
              type="button"
              class="q-btn"
              :class="{ active: quickSelectAllActive }"
              @click="onSelectAll"
            >
              全选
            </button>
            <button
              type="button"
              class="q-btn"
              :class="{ active: quickDomesticActive }"
              @click="onSelectDomestic"
            >
              国产
            </button>
          </div>
          <div class="brand-list">
            <div
              v-for="p in PLATFORMS"
              :key="p.key"
              class="brand-item"
              :class="{ active: selectedPlatKeys.includes(p.key) }"
              role="button"
              tabindex="0"
              @click="onTogglePlat(p.key)"
              @keydown.enter.prevent="onTogglePlat(p.key)"
              @keydown.space.prevent="onTogglePlat(p.key)"
            >
              <div class="brand-icon-wrap" aria-hidden="true">
                <img class="brand-icon-img" :src="platIconSrc(p.key)" alt="" loading="lazy" />
              </div>
              <div class="brand-text">
                <div class="brand-name">{{ platLineName(p) }}</div>
                <div class="brand-tagline">{{ platTagline(p) }}</div>
              </div>
              <div class="brand-radio" aria-hidden="true" />
            </div>
          </div>
        </div>
        <div class="sb-section">
          <div class="sb-section-head">
            <span class="sb-section-title">测试维度</span>
            <div class="sb-section-line" />
          </div>
          <div class="dim-grid">
            <div
              v-for="(d, i) in DIMS"
              :key="d.key"
              class="dim-item"
              :class="{ active: activeDim === i }"
              role="button"
              tabindex="0"
              @click="onSetDim(i)"
              @keydown.enter.prevent="onSetDim(i)"
              @keydown.space.prevent="onSetDim(i)"
            >
              {{ d.label }}
            </div>
          </div>
          <div class="sidebar-foot">
            数据更新于 {{ dataUpdatedAtDisplay }}
          </div>
        </div>
      </div>
    </div>
  </aside>
</template>
