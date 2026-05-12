<script setup lang="ts">
import { computed } from 'vue'
import { BENCHMARK_DATA_META, DATA_UPDATED_AT, OP_TABLE } from '@/data'
import {
  formatOperatorCsvDateDisplay,
  maxOperatorCsvDateForPlatform,
  type OperatorTableRow,
} from '@/features/dashboard/operatorBenchmark'
import { useInfiniDashboard } from '@/composables/useInfiniDashboard'
import { useDashboardNavigation } from '@/composables/useDashboardNavigation'

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
const { goOverview } = useDashboardNavigation()

/** public 下文件名与 PLATFORMS.key 对应（Vite BASE_URL 兼容子路径部署） */
const PLAT_ICON_FILE: Record<string, string> = {
  nvidia: 'nvidia.png',
  mthreads: 'moore.png',
  cambricon: 'cambricon.png',
  metax: 'metax.png',
  iluvatar: 'iluvatar.png',
  ascend: 'ascend.png',
  hygon: 'hygon.png',
  generic: 'ali.png',
}

function platIconSrc(key: string) {
  const file = PLAT_ICON_FILE[key]
  if (!file) return ''
  const b = import.meta.env.BASE_URL
  return b.endsWith('/') ? `${b}${file}` : `${b}/${file}`
}

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

function isBrandActive(key: string) {
  return selectedPlatKeys.value.includes(key)
}

/** 侧栏主文案与设计稿一致（不改 PLATFORMS 数据，仅展示层） */
function platLineName(p: (typeof PLATFORMS)[number]) {
  if (p.key === 'nvidia') return 'NVIDIA'
  if (p.key === 'generic') return '阿里'
  return p.name
}

/** 算子维度：侧栏日期来自 CSV `date`（详情=当前平台行内最大；概览=生成数据写入的算子集最大日期） */
const dataUpdatedAtDisplay = computed(() => {
  if (DIMS[activeDim.value]?.key !== 'op') return DATA_UPDATED_AT
  const meta = (BENCHMARK_DATA_META as { opDatasetUpdatedAt?: string }).opDatasetUpdatedAt
  if (currentView.value === 'detail') {
    const plat = detailState.value.platKey
    const tbl = OP_TABLE as unknown as Record<string, Record<string, OperatorTableRow[]>>
    const maxRow = maxOperatorCsvDateForPlatform(tbl[plat])
    if (maxRow) return formatOperatorCsvDateDisplay(maxRow)
  }
  if (meta) return meta
  return DATA_UPDATED_AT
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
              @click="selectAll"
            >
              全选
            </button>
            <button
              type="button"
              class="q-btn"
              :class="{ active: quickDomesticActive }"
              @click="selectDomestic"
            >
              国产
            </button>
          </div>
          <div class="brand-list">
            <div
              v-for="p in PLATFORMS"
              :key="p.key"
              class="brand-item"
              :class="{ active: isBrandActive(p.key) }"
              role="button"
              tabindex="0"
              @click="togglePlat(p.key)"
              @keydown.enter.prevent="togglePlat(p.key)"
              @keydown.space.prevent="togglePlat(p.key)"
            >
              <div class="brand-icon-wrap" aria-hidden="true">
                <img class="brand-icon-img" :src="platIconSrc(p.key)" alt="" loading="lazy" />
              </div>
              <div class="brand-text">
                <div class="brand-name">{{ platLineName(p) }}</div>
                <div class="brand-type">{{ p.type }}</div>
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
