<script setup lang="ts">
import { computed } from 'vue'
import type { CardRow } from '@/features/dashboard/dashboardFilterHelpers'
import { useInfiniDashboard } from '@/composables/useInfiniDashboard'

const {
  PLATFORMS,
  overviewCards,
  comparePlatKeys,
  openDetail,
  toggleCompare,
} = useInfiniDashboard()

const sortedCards = computed(() => overviewCards.value)

function platOf(card: CardRow) {
  return PLATFORMS.find((p) => p.key === card.key)!
}

function scoreColor(card: CardRow) {
  const s = card.ownScore ?? 0
  return s >= 100 ? '#2e7d32' : s >= 60 ? '#e65100' : '#c62828'
}

function inCompare(key: string) {
  return comparePlatKeys.value.includes(key)
}
</script>

<template>
  <div class="cards-grid">
    <div
      v-if="!sortedCards.length"
      style="
        grid-column: 1/-1;
        text-align: center;
        padding: 50px;
        color: #aaa;
        font-size: 14px;
        background: #fff;
        border-radius: var(--card-radius);
      "
    >
      暂无该维度数据，请切换平台或维度
    </div>
    <div
      v-for="c in sortedCards"
      v-else
      :key="c.key"
      class="score-card"
      @click="openDetail(c.key)"
    >
      <div class="card-head">
        <div class="card-brand-wrap">
          <div class="card-logo" :style="{ background: platOf(c).color }">{{ platOf(c).logo }}</div>
          <div>
            <div class="card-brand-name">{{ platOf(c).name }}</div>
            <span
              class="card-type-badge"
              :class="platOf(c).domestic ? 'badge-dom' : 'badge-intl'"
            >{{ platOf(c).type }}</span>
          </div>
        </div>
      </div>
      <div class="card-scores">
        <div class="score-col own">
          <div class="sc-label" :style="{ color: platOf(c).color }">{{ c.ownFw }}</div>
          <div class="sc-num" :style="{ color: scoreColor(c) }">{{ c.ownScore }}</div>
          <div class="sc-val" :style="{ color: platOf(c).color }">{{ c.ownVal }}</div>
        </div>
        <div v-if="c.openScore != null" class="score-col base">
          <div class="sc-label">{{ c.openFw }}</div>
          <div class="sc-num" style="color: #999">{{ c.openScore }}</div>
          <div class="sc-val" style="color: #aaa">{{ c.openVal || '—' }}</div>
        </div>
        <div
          v-else
          class="score-col base"
          style="display: flex; align-items: center; justify-content: center"
        >
          <span style="color: #ccc; font-size: 12px; font-style: italic">待测</span>
        </div>
      </div>
      <div class="card-stats">
        <div class="stat-item">
          <div class="stat-lbl">测试条数</div>
          <div class="stat-val">{{ c.n }}</div>
        </div>
        <div class="stat-item">
          <div class="stat-lbl">配置</div>
          <div class="stat-val" style="font-size: 11px">{{ c.extra }}</div>
        </div>
        <div class="stat-item">
          <div class="stat-lbl">得分</div>
          <div class="stat-val" :style="{ color: scoreColor(c) }">{{ c.ownScore }}</div>
        </div>
      </div>
      <div class="card-foot">
        <div class="perf-badge" :class="{ bad: !c.adv }">{{ c.advTxt }}</div>
        <div class="card-actions">
          <button
            type="button"
            class="mini-btn"
            @click.stop="openDetail(c.key)"
          >
            详情
          </button>
          <button
            type="button"
            class="mini-btn"
            :class="{ selected: inCompare(c.key) }"
            @click.stop="toggleCompare(c.key)"
          >
            {{ inCompare(c.key) ? '已加入' : '加入对比' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>
