<script setup lang="ts">
import { computed } from 'vue'
import type { ColumnsType } from 'ant-design-vue/es/table'
import VChart from 'vue-echarts'
import { PLATFORMS } from '@/data'
import type { CardRow } from '@/features/dashboard/dashboardFilterHelpers'
import { useInfiniDashboard } from '@/composables/useInfiniDashboard'
import { useDashboardNavigation } from '@/composables/useDashboardNavigation'
import { SCORE_TIER_COLOR } from '@/utils/scoreColor'
import { DETAIL_DUAL_BAR_PRIMARY, DETAIL_DUAL_BAR_SECONDARY } from '@/utils/echartsInfini'

type CompareTableRecord = {
  plat: (typeof PLATFORMS)[number]
  card: CardRow
  delta: number | null
}

/** 列宽总和与 scroll.x 一致（与详情页数据明细 a-table 用法对齐） */
const compareTableColumns: ColumnsType<CompareTableRecord> = [
  { title: '平台', key: 'plat', width: 168, ellipsis: true },
  { title: '类型', key: 'platType', width: 96, ellipsis: true },
  { title: '平均得分', key: 'ownScore', width: 104 },
  { title: '自研代表值', key: 'ownVal', minWidth: 120, width: 132, ellipsis: true },
  { title: '开源/参考值', key: 'openVal', width: 132, ellipsis: true },
  { title: '测试条数', key: 'n', width: 96 },
  { title: '配置', key: 'extra', minWidth: 140, width: 200, ellipsis: true },
  { title: '相对 NVIDIA', key: 'delta', width: 112 },
]

function compareRowKey(r: CompareTableRecord) {
  return r.plat.key
}

function compareDeltaText(delta: number | null) {
  if (delta == null) return '—'
  if (delta >= 0) return `+${delta}%`
  return `${delta}%`
}

const {
  comparePageTitle,
  comparePageSubtitle,
  compareKpiBlocks,
  compareScoreOption,
  compareLatencyOption,
  compareTableRows,
} = useInfiniDashboard()
const { goOverview } = useDashboardNavigation()

function hasChart(opt: object) {
  return opt && Object.keys(opt).length > 0
}

const showLatencyChart = computed(
  () => hasChart(compareLatencyOption.value as object),
)
</script>

<template>
  <div class="compare-panel">
    <header class="compare-panel__header">
      <div class="detail-title-row">
        <div>
          <div class="detail-title">{{ comparePageTitle }}</div>
          <div class="compare-subtitle">{{ comparePageSubtitle }}</div>
        </div>
        <a-button type="primary" @click="goOverview">返回概览</a-button>
      </div>
    </header>
    <div class="compare-panel__scroll">
      <div class="compare-kpi-grid">
        <div
          v-for="row in compareKpiBlocks"
          :key="row.plat.key"
          class="compare-kpi-card"
          :style="
            row.isBest
              ? {
                  border: `2px solid ${SCORE_TIER_COLOR.high}`,
                  background: '#F2F3F5',
                }
              : {}
          "
        >
          <div class="compare-kpi-name">
            <span class="compare-dot" :style="{ background: row.plat.color }" />
            {{ row.plat.name }}
            <span
              v-if="row.isBest"
              :style="{
                marginLeft: 'auto',
                fontSize: '10px',
                background: SCORE_TIER_COLOR.high,
                color: '#fff',
                padding: '2px 7px',
                borderRadius: '10px',
              }"
            >最高</span>
          </div>
          <div class="compare-kpi-score" :style="{ color: row.plat.color }">
            {{ row.card.ownScore ?? '—' }}
          </div>
          <div class="compare-kpi-meta">
            {{ row.card.ownVal || '—' }} · {{ row.card.extra || '' }} · 排名 {{ row.rank }}
          </div>
        </div>
      </div>

    <div class="charts-grid">
      <div class="chart-card">
        <div class="chart-title">自研提速倍率对比</div>
        <div style="font-size: 11px; color: #aaa; margin: -10px 0 12px">
          相对开源基准的提速倍率 · 灰线为基准×1.0 · 越高越好
        </div>
        <v-chart
          v-if="hasChart(compareScoreOption)"
          class="compare-chart"
          :option="compareScoreOption"
          autoresize
        />
      </div>
      <div class="chart-card">
        <div class="chart-title">代表延迟对比（ms，越低越好）</div>
        <div style="font-size: 11px; color: #aaa; margin: -10px 0 12px">
          各平台代表算子延迟值 · 单位 ms · 越低越好
        </div>
        <v-chart
          v-if="showLatencyChart"
          class="compare-chart"
          :option="compareLatencyOption"
          autoresize
        />
      </div>
    </div>

    <div class="table-card">
      <div class="table-head-row">
        <div class="table-title">对比明细</div>
        <div style="font-size: 12px; color: #888">数值来自当前维度卡片数据</div>
      </div>
      <div class="table-wrap">
        <a-table
          v-if="compareTableRows.length"
          :columns="compareTableColumns"
          :data-source="(compareTableRows as CompareTableRecord[])"
          :pagination="false"
          :bordered="false"
          table-layout="fixed"
          :scroll="{ x: 1040 }"
          :row-key="compareRowKey"
        >
          <template #bodyCell="{ column, record }">
            <template v-if="column.key === 'plat'">
              <span
                class="compare-dot"
                style="display: inline-block; margin-right: 8px; vertical-align: middle"
                :style="{ background: record.plat.color }"
              />
              {{ record.plat.name }}
            </template>
            <template v-else-if="column.key === 'platType'">
              {{ record.plat.type }}
            </template>
            <template v-else-if="column.key === 'ownScore'">
              <span class="score-num" :style="{ color: record.plat.color, fontWeight: 700 }">
                {{ record.card.ownScore ?? '—' }}
              </span>
            </template>
            <template v-else-if="column.key === 'ownVal'">
              <span :style="{ color: DETAIL_DUAL_BAR_PRIMARY, fontWeight: 700 }">{{ record.card.ownVal || '—' }}</span>
            </template>
            <template v-else-if="column.key === 'openVal'">
              <span :style="{ color: DETAIL_DUAL_BAR_SECONDARY, fontWeight: 700 }">{{ record.card.openVal || '—' }}</span>
            </template>
            <template v-else-if="column.key === 'n'">
              {{ record.card.n ?? '—' }}
            </template>
            <template v-else-if="column.key === 'extra'">
              {{ record.card.extra || '—' }}
            </template>
            <template v-else-if="column.key === 'delta'">
              <span
                v-if="record.delta != null"
                :class="record.delta >= 0 ? 'score-up' : 'score-dn'"
                class="score-num"
              >
                {{ compareDeltaText(record.delta) }}
              </span>
              <span v-else class="score-num score-na">—</span>
            </template>
          </template>
        </a-table>
      </div>
    </div>
    </div>
  </div>
</template>

<style scoped>
.compare-panel {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
  height: 100%;
  background-color: #fff;
}
.compare-panel__header {
  flex-shrink: 0;
}
.compare-panel__scroll {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  overflow-x: hidden;
  padding-bottom: 4px;
  background-color: #fff;
}
.compare-panel__scroll .compare-kpi-grid {
  margin-bottom: 16px;
}
.compare-panel__scroll :deep(.ant-table-thead > tr > th.ant-table-cell),
.compare-panel__scroll :deep(.ant-table-tbody > tr > td.ant-table-cell) {
  text-align: left !important;
}
.compare-chart {
  height: 240px;
  width: 100%;
  display: block;
  line-height: 0;
}
.charts-grid .chart-card {
  padding-bottom: 0;
}
</style>
