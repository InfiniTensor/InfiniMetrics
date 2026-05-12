<script setup lang="ts">
import { MinusOutlined, PlusOutlined } from '@ant-design/icons-vue'
import { computed } from 'vue'
import type { CardRow } from '@/features/dashboard/dashboardFilterHelpers'
import { useInfiniDashboard } from '@/composables/useInfiniDashboard'
import { useDashboardNavigation } from '@/composables/useDashboardNavigation'

const { PLATFORMS, overviewCards, comparePlatKeys, toggleCompare } = useInfiniDashboard()
const { goDetail } = useDashboardNavigation()

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

/** 拆分 advTxt：带 % 的数字片段加粗（如 1316%、10%），避免匹配 A100 等非百分比数字 */
function advTxtSegments(text: string): { text: string; bold: boolean }[] {
  const re = /\d+(?:\.\d+)?%/g
  const out: { text: string; bold: boolean }[] = []
  let pos = 0
  let m: RegExpExecArray | null
  while ((m = re.exec(text)) !== null) {
    if (m.index > pos) {
      out.push({ text: text.slice(pos, m.index), bold: false })
    }
    out.push({ text: m[0], bold: true })
    pos = m.index + m[0].length
  }
  if (pos < text.length) {
    out.push({ text: text.slice(pos), bold: false })
  }
  return out.length ? out : [{ text, bold: false }]
}
</script>

<template>
  <div class="dashboard-overview-cards">
    <div v-if="!sortedCards.length" class="dashboard-overview-cards__empty">
      暂无该维度数据，请切换平台或维度
    </div>
    <div
      v-for="c in sortedCards"
      v-else
      :key="c.key"
      class="score-card"
      @click="goDetail(c.key)"
    >
      <div class="card-head">
        <div class="card-head-row card-head-row--top">
          <div class="card-logo" :style="{ background: platOf(c).color }">{{ platOf(c).logo }}</div>
          <div class="card-brand-name">{{ platOf(c).name }}</div>
          <a-button
            type="text"
            size="small"
            class="card-head-compare-btn"
            :class="{ 'card-head-compare-btn--in': inCompare(c.key) }"
            :aria-label="inCompare(c.key) ? '移除对比' : '加入对比'"
            :title="inCompare(c.key) ? '移除对比' : '加入对比'"
            @click.stop="toggleCompare(c.key)"
          >
            <span class="card-head-compare-btn__inner">
              <PlusOutlined v-if="!inCompare(c.key)" class="card-head-compare-btn__ico" />
              <MinusOutlined v-else class="card-head-compare-btn__ico" />
              <span class="card-head-compare-btn__txt">{{
                inCompare(c.key) ? '移除对比' : '加入对比'
              }}</span>
            </span>
          </a-button>
        </div>
        <div class="card-head-row card-head-row--tag">
          <a-tag
            size="small"
            :color="platOf(c).domestic ? 'green' : 'blue'"
          >
            {{ platOf(c).type }}
          </a-tag>
          <a-tag
            v-if="c.advTxt"
            size="small"
            :color="c.adv ? 'green' : 'red'"
          >
            <template v-for="(seg, idx) in advTxtSegments(c.advTxt)" :key="idx">
              <strong v-if="seg.bold" class="adv-txt-num">{{ seg.text }}</strong>
              <template v-else>{{ seg.text }}</template>
            </template>
          </a-tag>
        </div>
      </div>
      <div class="card-scores">
        <div class="score-col own">
          <div class="sc-label" :style="{ color: platOf(c).color }">{{ c.ownFw }}</div>
          <div class="sc-num" :style="{ color: scoreColor(c) }">{{ c.ownScore }}</div>
          <div v-if="c.inferOwnCaption" class="sc-cap">{{ c.inferOwnCaption }}</div>
          <div class="sc-val" :style="{ color: platOf(c).color }">{{ c.ownVal }}</div>
        </div>
        <div v-if="c.openScore != null" class="score-col base">
          <div class="sc-label sc-label--base">{{ c.openFw }}</div>
          <div class="sc-num sc-num--base">{{ c.openScore }}</div>
          <div v-if="c.inferOpenCaption" class="sc-cap sc-cap--base">{{ c.inferOpenCaption }}</div>
          <div class="sc-val sc-val--base">{{ c.openVal || '—' }}</div>
        </div>
        <div
          v-else
          class="score-col base"
          style="display: flex; align-items: center; justify-content: center"
        >
          <span class="sc-base-pending">待测</span>
        </div>
      </div>
      <div class="card-stats">
        <div class="stat-item">
          <div class="stat-lbl">测试条数</div>
          <div class="stat-val">{{ c.n }}</div>
        </div>
        <div class="stat-item">
          <div class="stat-lbl">配置</div>
          <div class="stat-val">{{ c.extra }}</div>
        </div>
        <div class="stat-item">
          <div class="stat-lbl">得分</div>
          <div class="stat-val" :style="{ color: scoreColor(c) }">{{ c.ownScore }}</div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* 概览平台卡片：扁平描边容器（业务逻辑未改，仅 UI） */
.dashboard-overview-cards {
  --overview-card-border: #e5e8ef;
  --overview-card-radius: 4px;

  display: grid;
  width: 100%;
  box-sizing: border-box;
  /* 每行三列均分剩余宽度；minmax(0,1fr) 防止长内容撑破网格 */
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
}

.dashboard-overview-cards__empty {
  grid-column: 1 / -1;
  text-align: center;
  padding: 50px;
  color: #aaa;
  font-size: 14px;
  background: #fff;
  border-radius: var(--overview-card-radius);
  border: 1px solid var(--overview-card-border);
  box-shadow: none;
}

.score-card {
  font-size: 14px;
  background: #fff;
  border-radius: var(--overview-card-radius);
  border: 1px solid var(--overview-card-border);
  padding: 22px;
  box-shadow: none;
  cursor: pointer;
  transition:
    border-color 0.2s ease,
    background-color 0.2s ease;
  position: relative;
  overflow: hidden;
}

.score-card:hover {
  transform: none;
  box-shadow: none;
  background: #fafbfc;
  border-color: var(--color-primary);
}

.score-card::before {
  display: none;
}

.card-head {
  --card-logo-size: 32px;
  --card-head-inline-gap: 10px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  margin-bottom: 18px;
}

.card-head-row--top {
  display: flex;
  align-items: center;
  gap: var(--card-head-inline-gap);
  min-width: 0;
}

.card-head-compare-btn {
  flex-shrink: 0;
  margin-left: auto;
}

/* type="text"：不包 Wave，去掉点击水波纹/扩散动画；样式仍按原默认按钮意图覆盖 */
.card-head-compare-btn.ant-btn.ant-btn-text {
  border-radius: 2px;
  border: none;
  box-shadow: none;
  font-size: 12px;
  color: #4e5969;
  background: transparent;
  padding-inline: 8px;
  height: auto;
  transition: none;
}

/* 未加入：hover / 键盘聚焦时浅灰底（不含「移除对比」绿底状态） */
.card-head-compare-btn.ant-btn:not(.card-head-compare-btn--in):hover,
.card-head-compare-btn.ant-btn:not(.card-head-compare-btn--in):focus-visible {
  border: none;
  box-shadow: none;
  color: #4e5969;
  background: #f2f3f5;
}

/* 仅「有鼠标焦点但鼠标未悬停」时清掉底色；否则与 :hover 同时命中时透明会盖掉浅灰 */
.card-head-compare-btn.ant-btn:not(.card-head-compare-btn--in):focus:not(:focus-visible):not(:hover) {
  border: none;
  box-shadow: none;
  color: #4e5969;
  background: transparent;
}

.card-head-compare-btn__inner {
  display: inline-flex;
  align-items: center;
  justify-content: flex-end;
  gap: 4px;
}

.card-head-compare-btn__txt {
  color: inherit;
}

.card-head-compare-btn__ico {
  font-size: 12px;
  color: inherit;
}

/* 与 .ant-btn-text 同优先级，保证「移除对比」浅绿底不被 transparent 盖掉 */
.card-head-compare-btn--in.ant-btn.ant-btn-text {
  color: #4e5969;
  background: #e8f5e9;
}

.card-head-compare-btn--in.ant-btn.ant-btn-text:hover,
.card-head-compare-btn--in.ant-btn.ant-btn-text:focus-visible {
  color: #4e5969;
  background: #c8e6c9;
}

.card-head-compare-btn--in.ant-btn.ant-btn-text:focus:not(:focus-visible):not(:hover) {
  color: #4e5969;
  background: #e8f5e9;
}

/* 第二行：国产/国际与「自研快」等并列，与头像左对齐；圆角 6px */
.card-head-row--tag {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

.card-head-row--tag :deep(.ant-tag) {
  border-radius: 6px;
  font-size: 14px !important;
}

.adv-txt-num {
  font-weight: 700;
}

.card-logo {
  width: var(--card-logo-size);
  height: var(--card-logo-size);
  flex-shrink: 0;
  border-radius: 7px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #fff;
  font-size: 14px;
  font-weight: 700;
}

.card-brand-name {
  flex: 1;
  font-size: 14px;
  font-weight: 700;
  color: #1a1a2e;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.card-scores {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  margin-bottom: 16px;
}

.score-col {
  text-align: center;
  padding: 14px 10px;
  border-radius: 4px;
  background: #f2f3f5;
}

.sc-label {
  font-size: 14px;
  color: #888;
  margin-bottom: 6px;
}

.sc-num {
  font-size: 14px;
  font-weight: 700;
  margin-bottom: 2px;
}

.sc-cap {
  font-size: 12px;
  color: #888;
  line-height: 1.2;
  margin-bottom: 4px;
}

.sc-cap--base {
  color: #8c8c8c;
}

.score-col.own .sc-num {
  color: var(--blue, #5b7ec9);
}

/* 灰底列：字色加深，保证在 #F2F3F5 上可读 */
.sc-label.sc-label--base {
  color: #595959;
}

.sc-num.sc-num--base {
  color: #262626;
}

.sc-val.sc-val--base {
  color: #595959;
}

.sc-base-pending {
  color: #8c8c8c;
  font-style: italic;
}

.sc-val {
  font-size: 14px;
  color: #666;
}

.card-stats {
  display: flex;
  justify-content: space-between;
  padding: 0;
  margin: 0;
  background: transparent;
  border: none;
  box-shadow: none;
}

.stat-item {
  text-align: center;
}

.stat-lbl {
  font-size: 14px;
  color: #aaa;
  margin-bottom: 3px;
}

.stat-val {
  font-size: 14px;
  font-weight: 600;
  color: #1a1a2e;
}

@media (max-width: 1024px) {
  .dashboard-overview-cards {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 700px) {
  .dashboard-overview-cards {
    grid-template-columns: minmax(0, 1fr);
  }

  .card-scores {
    grid-template-columns: 1fr;
  }
}
</style>
