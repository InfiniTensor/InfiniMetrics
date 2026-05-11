<script setup lang="ts">
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { useInfiniDashboard } from '@/composables/useInfiniDashboard'
import { useDashboardNavigation } from '@/composables/useDashboardNavigation'

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
  <div>
    <div class="compare-page-head">
      <div class="compare-title-row">
        <div>
          <div class="compare-page-title">{{ comparePageTitle }}</div>
          <div class="compare-subtitle">{{ comparePageSubtitle }}</div>
        </div>
        <button type="button" class="back-btn" @click="goOverview">← 返回概览</button>
      </div>
      <div class="compare-kpi-grid">
        <div
          v-for="row in compareKpiBlocks"
          :key="row.plat.key"
          class="compare-kpi-card"
          :style="
            row.isBest
              ? {
                  border: '2px solid #2e7d32',
                  background: 'linear-gradient(135deg,#e8f5e9,#f1f8f1)',
                }
              : {}
          "
        >
          <div class="compare-kpi-name">
            <span class="compare-dot" :style="{ background: row.plat.color }" />
            {{ row.plat.name }}
            <span
              v-if="row.isBest"
              style="
                margin-left: auto;
                font-size: 10px;
                background: #2e7d32;
                color: #fff;
                padding: 2px 7px;
                border-radius: 10px;
              "
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
        <table>
          <thead>
            <tr>
              <th>平台</th>
              <th>类型</th>
              <th>平均得分</th>
              <th>自研代表值</th>
              <th>开源/参考值</th>
              <th>测试条数</th>
              <th>配置</th>
              <th>相对 A100</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in compareTableRows" :key="row.plat.key">
              <td>
                <span
                  class="compare-dot"
                  style="display: inline-block; margin-right: 8px"
                  :style="{ background: row.plat.color }"
                />
                {{ row.plat.name }}
              </td>
              <td>{{ row.plat.type }}</td>
              <td style="font-weight: 700" :style="{ color: row.plat.color }">
                {{ row.card.ownScore ?? '—' }}
              </td>
              <td>{{ row.card.ownVal || '—' }}</td>
              <td>{{ row.card.openVal || '—' }}</td>
              <td>{{ row.card.n ?? '—' }}</td>
              <td>{{ row.card.extra || '—' }}</td>
              <td>
                {{
                  row.delta == null
                    ? '—'
                    : row.delta >= 0
                      ? `+${row.delta}%`
                      : `${row.delta}%`
                }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<style scoped>
.compare-chart {
  height: 260px;
  width: 100%;
}
</style>
