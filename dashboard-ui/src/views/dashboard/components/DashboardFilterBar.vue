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

/** 筛选区（含已选对比）Ant Design 主题：与设计稿主色、圆角一致 */
const filterBarTheme = {
  token: {
    colorPrimary: 'hsl(228.75deg 22.64% 58.43%)',
    borderRadius: 6,
    borderRadiusLG: 6,
    borderRadiusSM: 6,
    fontSizeSM: 16,
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
          <a-space :size="[6, 6]" wrap>
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
            :title="sortDesc ? '得分：高 → 低（点击切换为升序）' : '得分：低 → 高（点击切换为降序）'"
            @click="toggleSort"
          >
            <span class="filter-sort-arrows" aria-hidden="true">
              <span class="filter-sort-arrows__up"></span>
              <span class="filter-sort-arrows__down"></span>
            </span>
            <span class="filter-sort-text">按得分</span>
            <span class="filter-sort-caret" aria-hidden="true"></span>
          </a-button>
        </div>
      </div>

      <!-- 概览 / 对比页展示；详情页不展示（业务上详情无「已选对比」） -->
      <div v-if="currentView !== 'detail'" class="compare-bar">
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
        <div class="compare-bar-actions">
          <div class="compare-bar-actions-btns">
            <a-button size="small" @click="onClearCompare">清空</a-button>
            <a-button type="primary" size="small" @click="onOpenComparePage">开始对比</a-button>
          </div>
        </div>
      </div>
    </div>
  </div>
  </a-config-provider>
</template>

<style scoped>
.filter-bar-root {
  --filter-bar-font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei',
    sans-serif;
  --fb-active-bg: hsl(228.75deg 22.64% 58.43%);
  --fb-active-bg-hover: hsl(228.75deg 22.64% 51%);
  --fb-inactive-text: hsl(228.84deg 22.99% 63.33%);
  --fb-label: #333e5a;
  --fb-muted-border: rgba(51, 62, 90, 0.14);
  width: 100%;
  min-width: 0;
  box-sizing: border-box;
  background: var(--page-bg);
  border-radius: 0;
  box-shadow: none;
  font-family: var(--filter-bar-font-family);
}

.filter-bar-root :deep(.ant-btn),
.filter-bar-root :deep(.ant-btn .ant-btn-text),
.filter-bar-root :deep(.ant-tag) {
  font-family: var(--filter-bar-font-family);
}

.filter-bar-body {
  --filter-label-col: 92px;
  --filter-label-gap: 10px;
  display: grid;
  grid-template-columns: var(--filter-label-col) minmax(0, 1fr) auto;
  column-gap: var(--filter-label-gap);
  row-gap: 12px;
  align-items: start;
  width: 100%;
  min-width: 0;
}

/* 子行参与父级三列网格，第三列宽度取各行最宽（与「清空+开始对比」总宽对齐后，「按得分」拉满） */
.filter-bar-root .filter-row {
  display: contents;
}

.filter-bar-root .filter-row + .filter-row {
  margin-top: 0;
}

.filter-bar-root .filter-label {
  width: 100%;
  min-width: 0;
  max-width: var(--filter-label-col);
  font-size: 16px;
  font-weight: 600;
  color: rgb(32 39 75);
  flex-shrink: 0;
  text-align: left;
}

.filter-bar-root .filter-pills {
  flex: unset;
  min-width: 0;
}

.filter-row-aside {
  display: flex;
  justify-content: flex-start;
  align-items: flex-start;
  min-width: 0;
  width: 100%;
}

/* 排序：白底圆角，类下拉样式 */
.filter-sort-btn.filter-sort-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  width: 100%;
  min-width: 0;
  box-sizing: border-box;
  height: 32px;
  padding: 0 14px;
  border-radius: 6px;
  font-size: var(--filter-bar-font-sm);
  font-weight: 500;
  color: var(--fb-label);
  background: #fff;
  border: 1px solid var(--fb-muted-border);
  box-shadow: none;
}

.filter-sort-btn:hover,
.filter-sort-btn:focus {
  color: var(--fb-label);
  border-color: rgba(51, 62, 90, 0.24);
  background: #fff;
}

/* 左侧：上下实心小三角（排序）；右侧：实心下三角（下拉示意） */
.filter-sort-arrows {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 2px;
  flex-shrink: 0;
  width: 10px;
  height: 14px;
  opacity: 0.88;
  color: currentColor;
}

.filter-sort-arrows__up {
  width: 0;
  height: 0;
  border-left: 3.5px solid transparent;
  border-right: 3.5px solid transparent;
  border-bottom: 4.5px solid currentColor;
}

.filter-sort-arrows__down {
  width: 0;
  height: 0;
  border-left: 3.5px solid transparent;
  border-right: 3.5px solid transparent;
  border-top: 4.5px solid currentColor;
}

.filter-sort-text {
  flex: 1;
  text-align: center;
  min-width: 3em;
}

.filter-sort-caret {
  flex-shrink: 0;
  width: 0;
  height: 0;
  border-left: 4px solid transparent;
  border-right: 4px solid transparent;
  border-top: 5px solid currentColor;
  opacity: 0.88;
  margin-top: 1px;
}

/* 已选对比行：与筛选行共用 filter-bar-body 三列网格（display:contents） */
.filter-bar-root .compare-bar {
  display: contents;
}

.filter-bar-root .compare-title {
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
}

.filter-bar-root .compare-empty {
  font-size: var(--filter-bar-font-sm);
  color: rgba(51, 62, 90, 0.45);
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.filter-bar-root .compare-bar-actions {
  margin-left: 0;
  flex-shrink: 0;
  width: 100%;
  min-width: 0;
  box-sizing: border-box;
}

/* 两按钮同宽；整列宽度由本行撑开，与上行「按得分」同列对齐 */
.filter-bar-root .compare-bar-actions-btns {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  width: 100%;
  min-width: 0;
}

.filter-bar-root :deep(.compare-bar-actions-btns .ant-btn-sm) {
  width: 100%;
  min-width: 0;
}

/* 筛选 pill、对比区按钮：统一字重（覆盖 ant-btn-text 默认偏细） */
.filter-bar-root :deep(.filter-row .ant-btn-sm) {
  font-size: var(--filter-bar-font-sm);
  height: 28px;
  padding: 0 14px;
  line-height: 26px;
  border-radius: 6px;
  font-weight: 600;
}

.filter-bar-root :deep(.filter-row .ant-btn-sm .ant-btn-text) {
  font-weight: 600;
}

.filter-bar-root :deep(.filter-row .ant-btn-default.ant-btn-sm) {
  color: var(--fb-inactive-text);
  background: #fff;
  border-color: var(--fb-muted-border);
  box-shadow: none;
}

.filter-bar-root :deep(.filter-row .ant-btn-default.ant-btn-sm:hover) {
  color: var(--fb-inactive-text);
  border-color: rgba(51, 62, 90, 0.26);
  background: #fff;
}

.filter-bar-root :deep(.filter-row .ant-btn-primary.ant-btn-sm) {
  color: #fff;
  background: var(--fb-active-bg);
  border-color: var(--fb-active-bg);
  box-shadow: none;
}

.filter-bar-root :deep(.filter-row .ant-btn-primary.ant-btn-sm:hover) {
  color: #fff;
  background: var(--fb-active-bg-hover);
  border-color: var(--fb-active-bg-hover);
}

/* 对比区按钮与筛选 pill 视觉一致 */
.filter-bar-root :deep(.compare-bar-actions .ant-btn-sm) {
  font-size: var(--filter-bar-font-sm);
  height: 28px;
  padding: 0 16px;
  line-height: 26px;
  border-radius: 6px;
  font-weight: 600;
}

.filter-bar-root :deep(.compare-bar-actions .ant-btn-sm .ant-btn-text) {
  font-weight: 600;
}

.filter-bar-root :deep(.compare-bar-actions .ant-btn-default.ant-btn-sm) {
  color: var(--fb-inactive-text);
  background: #fff;
  border-color: var(--fb-muted-border);
  box-shadow: none;
}

.filter-bar-root :deep(.compare-bar-actions .ant-btn-default.ant-btn-sm:hover) {
  color: var(--fb-inactive-text);
  border-color: rgba(51, 62, 90, 0.26);
  background: #fff;
}

.filter-bar-root :deep(.compare-bar-actions .ant-btn-primary.ant-btn-sm) {
  color: #fff;
  background: var(--fb-active-bg);
  border-color: var(--fb-active-bg);
  box-shadow: none;
}

.filter-bar-root :deep(.compare-bar-actions .ant-btn-primary.ant-btn-sm:hover) {
  color: #fff;
  background: var(--fb-active-bg-hover);
  border-color: var(--fb-active-bg-hover);
}

/* 已选平台 Tag：白底 + 浅蓝灰字；字重与筛选项 pill 一致（ant-tag 默认偏细） */
.filter-bar-root :deep(.compare-chips .ant-tag) {
  margin-inline-end: 0;
  font-size: var(--filter-bar-font-sm);
  line-height: 24px;
  padding: 2px 10px;
  border-radius: 6px;
  color: var(--fb-inactive-text);
  background: #fff;
  border: 1px solid var(--fb-muted-border);
  font-weight: 600;
}

.filter-bar-root :deep(.compare-chips .ant-tag .ant-tag-close-icon) {
  font-weight: 400;
  color: rgba(51, 62, 90, 0.4);
}

.filter-bar-root :deep(.compare-chips .ant-tag .ant-tag-close-icon:hover) {
  color: var(--fb-label);
}

.filter-bar-root :deep(.compare-chips .ant-tag .anticon-close) {
  color: rgba(51, 62, 90, 0.4);
  font-weight: 400;
}

.filter-bar-root :deep(.compare-chips .ant-tag .anticon-close:hover) {
  color: var(--fb-label);
}
</style>
