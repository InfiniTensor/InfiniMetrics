<script setup lang="ts">
import { Modal } from 'ant-design-vue'
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useInfiniDashboard } from '@/composables/useInfiniDashboard'
import { useDashboardNavigation } from '@/composables/useDashboardNavigation'
import { OP_TABLE, INFER_TABLE, TRAIN_TABLE, COMM_TABLE, BW_TABLE, dtypesForOpPlatform } from '@/data'
import { bwPlatHasMode } from '@/features/dashboard/bwBenchmark'
import { commPlatHasCommType } from '@/features/dashboard/commBenchmark'
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
const router = useRouter()
const { goCompare, leaveCompareOrSyncOverview } = useDashboardNavigation()

/** 顶栏 pills：详情页只改筛选不跳路由；概览/对比仍走 leaveCompareOrSyncOverview */
function onSetFilter(fi: number, pi: number) {
  setFilter(fi, pi)
  if (router.currentRoute.value.name !== 'detail') {
    leaveCompareOrSyncOverview()
  }
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
const commTableForBar = COMM_TABLE as Record<string, { commType: string }[] | undefined>
const bwTableForBar = BW_TABLE as Parameters<typeof bwPlatHasMode>[1]

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
        pills: pills.filter((x) => x.label !== '全部' && batches.has(Number(x.label))),
      }
    }
    if (d.key === 'infer' && fi === 1 && view === 'detail') {
      const lens = inferNumericSetForPlatform(inferTableForBar, plat, 'inLen')
      return {
        label: f.label,
        pills: pills.filter((x) => x.label !== '全部' && lens.has(Number(x.label))),
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
        pills: pills.filter((x) => x.label !== '全部' && fwOnPlat.has(x.label.toLowerCase())),
      }
    }
    if (d.key === 'comm' && fi === 0 && view === 'detail') {
      return {
        label: f.label,
        pills: pills.filter(
          (x) => x.label !== '全部' && commPlatHasCommType(plat, commTableForBar, x.label),
        ),
      }
    }
    if (d.key === 'bw' && fi === 0 && view === 'detail') {
      return {
        label: f.label,
        pills: pills.filter((x) => x.label !== '全部' && bwPlatHasMode(plat, bwTableForBar, x.label)),
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

/** 筛选区（含已选对比）Ant Design 主题：圆角与字号仅影响外观 */
const filterBarTheme = {
  token: {
    colorPrimary: '#162b75',
    borderRadius: 10,
    borderRadiusLG: 10,
    borderRadiusSM: 10,
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
    <div class="filter-bar-body">
      <div
        v-for="(f, fi) in filterBarFilterRows"
        :key="f.label"
        class="filter-row"
      >
        <span class="filter-label">{{ f.label }}</span>
        <div class="filter-pills">
          <a-space :size="[8, 6]" wrap>
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
        <div class="filter-row-aside">
          <a-button
            v-if="fi === 0 && currentView === 'overview'"
            class="filter-sort-btn"
            size="small"
            @click="toggleSort"
          >
            <span class="filter-sort-lead" aria-hidden="true">⇅</span>
            <span class="filter-sort-text">按得分</span>
            <span class="filter-sort-dir" aria-hidden="true">{{ sortDesc ? '↓' : '↑' }}</span>
          </a-button>
        </div>
      </div>

      <!-- 概览 / 对比页展示；详情页不展示（业务上详情无「已选对比」） -->
      <div
        v-if="currentView !== 'detail'"
        class="compare-bar"
        :class="{ active: compareChosen.length > 0 }"
      >
        <div class="compare-bar-left">
          <div class="filter-label compare-title">已选对比</div>
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
        </div>
        <div class="compare-bar-actions">
          <a-space :size="8" :wrap="false">
            <a-button size="small" @click="onClearCompare">清空</a-button>
            <a-button type="primary" size="small" @click="onOpenComparePage">开始对比</a-button>
          </a-space>
        </div>
      </div>
    </div>
  </div>
  </a-config-provider>
</template>

<style scoped>
.filter-bar-root {
  width: 100%;
  min-width: 0;
  box-sizing: border-box;
}

.filter-bar-body {
  --filter-label-col: 76px;
  --filter-label-gap: 6px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  width: 100%;
  min-width: 0;
}

/* 三列：固定宽标签列，保证各行选项区左缘对齐 */
.filter-bar-root .filter-row {
  display: grid;
  grid-template-columns: var(--filter-label-col) minmax(0, 1fr) auto;
  align-items: center;
  column-gap: var(--filter-label-gap);
  row-gap: 8px;
  width: 100%;
  min-width: 0;
  margin-top: 0;
}

.filter-bar-root .filter-row + .filter-row {
  margin-top: 0;
}

.filter-bar-root .filter-label {
  width: 100%;
  min-width: 0;
  max-width: var(--filter-label-col);
  font-size: 14px;
  font-weight: 700;
  color: #000000;
  flex-shrink: 0;
  text-align: left;
}

.filter-bar-root .filter-pills {
  flex: unset;
  min-width: 0;
}

.filter-row-aside {
  display: flex;
  justify-content: flex-end;
  align-items: center;
  min-height: 28px;
}

/* 排序：白底圆角，类下拉样式 */
.filter-sort-btn.filter-sort-btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  height: 32px;
  padding: 0 14px;
  border-radius: 10px;
  font-size: var(--filter-bar-font-sm);
  font-weight: 500;
  color: #162b75;
  background: #fff;
  border: 1px solid rgba(22, 43, 117, 0.12);
  box-shadow: none;
}

.filter-sort-btn:hover,
.filter-sort-btn:focus {
  color: #162b75;
  border-color: rgba(22, 43, 117, 0.22);
  background: #fff;
}

.filter-sort-lead {
  font-size: 12px;
  line-height: 1;
  opacity: 0.85;
}

.filter-sort-text {
  flex: 1;
  text-align: center;
  min-width: 3em;
}

.filter-sort-dir {
  font-size: 12px;
  line-height: 1;
  opacity: 0.85;
}

/* 已选对比行：与筛选行同列宽，并覆盖全局 .compare-bar 的 display 规则 */
.filter-bar-root .compare-bar {
  display: none;
  margin-top: 0;
  width: 100%;
  min-width: 0;
  box-sizing: border-box;
}

.filter-bar-root .compare-bar.active {
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: space-between;
  gap: 12px 16px;
  flex-wrap: nowrap;
}

.compare-bar-left {
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: var(--filter-label-gap);
  min-width: 0;
  flex: 1;
}

.filter-bar-root .compare-title {
  flex: 0 0 var(--filter-label-col);
  width: var(--filter-label-col);
  min-width: var(--filter-label-col);
  max-width: var(--filter-label-col);
  white-space: nowrap;
  text-align: left;
}

.filter-bar-root .compare-chips {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  min-width: 0;
  flex: 1;
}

.filter-bar-root .compare-empty {
  font-size: 12px;
  color: #94a0b8;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.filter-bar-root .compare-bar-actions {
  margin-left: 0;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: flex-end;
}

/* 筛选 pill：未选中白底深蓝字；选中深蓝底白字 */
.filter-bar-root :deep(.filter-row .ant-btn-sm) {
  font-size: var(--filter-bar-font-sm);
  height: 28px;
  padding: 0 14px;
  line-height: 26px;
  border-radius: 10px;
  font-weight: 500;
}

.filter-bar-root :deep(.filter-row .ant-btn-default.ant-btn-sm) {
  color: #162b75;
  background: #fff;
  border-color: rgba(22, 43, 117, 0.14);
  box-shadow: none;
}

.filter-bar-root :deep(.filter-row .ant-btn-default.ant-btn-sm:hover) {
  color: #162b75;
  border-color: rgba(22, 43, 117, 0.28);
  background: #fff;
}

.filter-bar-root :deep(.filter-row .ant-btn-primary.ant-btn-sm) {
  color: #fff;
  background: #162b75;
  border-color: #162b75;
  box-shadow: none;
}

.filter-bar-root :deep(.filter-row .ant-btn-primary.ant-btn-sm:hover) {
  color: #fff;
  background: #1a3488;
  border-color: #1a3488;
}

/* 对比区按钮与筛选 pill 视觉一致 */
.filter-bar-root :deep(.compare-bar-actions .ant-btn-sm) {
  font-size: var(--filter-bar-font-sm);
  height: 28px;
  padding: 0 16px;
  line-height: 26px;
  border-radius: 10px;
  font-weight: 500;
}

.filter-bar-root :deep(.compare-bar-actions .ant-btn-default.ant-btn-sm) {
  color: #162b75;
  background: #fff;
  border-color: rgba(22, 43, 117, 0.14);
  box-shadow: none;
}

.filter-bar-root :deep(.compare-bar-actions .ant-btn-primary.ant-btn-sm) {
  color: #fff;
  background: #162b75;
  border-color: #162b75;
  box-shadow: none;
}

/* 已选平台 Tag：白底圆角深蓝字 */
.filter-bar-root :deep(.compare-chips .ant-tag) {
  margin-inline-end: 0;
  font-size: var(--filter-bar-font-sm);
  line-height: 22px;
  padding: 2px 10px;
  border-radius: 10px;
  color: #162b75;
  background: #fff;
  border: 1px solid rgba(22, 43, 117, 0.14);
}

.filter-bar-root :deep(.compare-chips .ant-tag .anticon-close) {
  color: rgba(22, 43, 117, 0.45);
}

.filter-bar-root :deep(.compare-chips .ant-tag .anticon-close:hover) {
  color: #162b75;
}
</style>
