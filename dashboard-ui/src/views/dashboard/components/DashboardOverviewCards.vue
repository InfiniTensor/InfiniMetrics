<script setup lang="ts">
import { MinusOutlined, PlusOutlined } from '@ant-design/icons-vue'
import { computed } from 'vue'
import type { CardRow } from '@/features/dashboard/dashboardFilterHelpers'
import { platIconSrc } from '@/features/dashboard/platPublicIcon'
import { useInfiniDashboard } from '@/composables/useInfiniDashboard'
import { useDashboardNavigation } from '@/composables/useDashboardNavigation'
import { scoreTierColor, scoreTierTagPreset } from '@/utils/scoreColor'

const { PLATFORMS, overviewCards, comparePlatKeys, toggleCompare } = useInfiniDashboard()
const { goDetail } = useDashboardNavigation()

const sortedCards = computed(() => overviewCards.value)

function platOf(card: CardRow) {
  return PLATFORMS.find((p) => p.key === card.key)!
}

/** NVIDIA 基准：仅保留「国际标杆」一条 tag，不展示 advTxt 第二条 */
function isNvidiaIntlOnlyOverviewTags(card: CardRow) {
  if (card.key.toLowerCase() === 'nvidia') return true
  const name = (platOf(card).name || '').trim().toLowerCase()
  return name === 'nvidia a100' || name === 'nvidia'
}

/** 概览卡自研列与 ownScoreColStyle 用 scoreTierColor；adv 标签高分档用 scoreTierTagPreset（蓝）。底部「得分」数值与两侧统计同色 */
function scoreColor(card: CardRow) {
  return scoreTierColor(card.ownScore)
}

/** 自研列（标签 / 得分 / 副标题 / 指标值）与得分同色，避免品牌色与分档色混用 */
function ownScoreColStyle(card: CardRow) {
  return { color: scoreColor(card) }
}

function scoreTagPreset(card: CardRow) {
  return scoreTierTagPreset(card.ownScore)
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
      :class="{ 'score-card--in-compare': inCompare(c.key) }"
      @click="goDetail(c.key)"
    >
      <div class="card-head">
        <div class="card-head-row card-head-row--main">
          <div class="card-logo" aria-hidden="true">
            <img
              v-if="platIconSrc(c.key)"
              class="card-logo-img"
              :src="platIconSrc(c.key)"
              alt=""
              loading="lazy"
            />
            <span
              v-else
              class="card-logo-fallback"
              :style="{ background: platOf(c).color }"
            >{{ platOf(c).logo }}</span>
          </div>
          <div class="card-head-center">
            <div class="card-brand-name">{{ platOf(c).name }}</div>
            <div class="card-head-tags">
              <a-tag
                size="small"
                :class="{ 'overview-tag--theme-blue': platOf(c).domestic }"
                :color="platOf(c).domestic ? 'blue' : 'green'"
              >
                {{ platOf(c).type }}
              </a-tag>
              <a-tag
                v-if="c.advTxt && !isNvidiaIntlOnlyOverviewTags(c)"
                size="small"
                :class="{ 'overview-tag--theme-blue': scoreTagPreset(c) === 'blue' }"
                :color="scoreTagPreset(c)"
              >
                <template v-for="(seg, idx) in advTxtSegments(c.advTxt)" :key="idx">
                  <strong v-if="seg.bold" class="adv-txt-num">{{ seg.text }}</strong>
                  <template v-else>{{ seg.text }}</template>
                </template>
              </a-tag>
            </div>
          </div>
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
      </div>
      <div
        class="card-scores"
        :class="{ 'card-scores--single': c.openScore == null }"
      >
        <div class="score-col own">
          <div class="sc-label" :style="ownScoreColStyle(c)">{{ c.ownFw }}</div>
          <div class="sc-num" :style="ownScoreColStyle(c)">{{
            c.ownScore != null ? c.ownScore : '—'
          }}</div>
          <div v-if="c.inferOwnCaption" class="sc-cap" :style="ownScoreColStyle(c)">{{
            c.inferOwnCaption
          }}</div>
          <div class="sc-val" :style="ownScoreColStyle(c)">{{ c.ownVal }}</div>
        </div>
        <div v-if="c.openScore != null" class="score-col base">
          <div class="sc-label sc-label--base">{{ c.openFw }}</div>
          <div class="sc-num sc-num--base">{{ c.openScore }}</div>
          <div v-if="c.inferOpenCaption" class="sc-cap sc-cap--base">{{ c.inferOpenCaption }}</div>
          <div class="sc-val sc-val--base">{{ c.openVal || '—' }}</div>
        </div>
      </div>
      <div class="card-stats">
        <div class="stat-item">
          <div class="stat-lbl">测试条数</div>
          <div class="stat-val">{{ c.n }}</div>
        </div>
        <div class="stat-item">
          <div class="stat-lbl">配置</div>
          <div class="stat-val stat-val--wrap">{{ c.extra }}</div>
        </div>
        <div class="stat-item">
          <div class="stat-lbl">得分</div>
          <div class="stat-val">{{
            c.footerScore != null ? c.footerScore : c.ownScore != null ? c.ownScore : '—'
          }}</div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* 概览平台卡片：浅灰底 + 白块对比区（仅 UI；中间自研列档色仍由 ownScoreColStyle 控制） */
.dashboard-overview-cards {
  --overview-card-border: #e8e8e8;
  --overview-card-border-in-compare: color-mix(in srgb, var(--blue) 45%, #d9d9d9);
  --overview-card-hover-bg: #e2e5ee;
  --overview-card-radius: 8px;
  --overview-muted: #7d89b0;
  /* 卡片内芯片/平台名称：略偏薰衣草灰蓝，与标签等 --overview-muted 区分 */
  --overview-brand-name: #8a94b7;
  --overview-caption: #9aa3bf;
  --overview-hover-title: #3d4a63;
  --overview-hover-tag-bg: #6b7389;
  --overview-hover-btn-bg: #5a6478;
  --overview-hover-btn-in-bg: #5c4f7a;

  display: grid;
  width: 100%;
  box-sizing: border-box;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  row-gap: 15px;
  column-gap: 22px;
}

.dashboard-overview-cards__empty {
  grid-column: 1 / -1;
  text-align: center;
  padding: 50px;
  color: var(--overview-caption);
  font-size: 14px;
  background: #f2f2f2;
  border-radius: var(--overview-card-radius);
  border: 1px solid var(--overview-card-border);
  box-shadow: none;
}

.score-card {
  font-size: 14px;
  background: #f2f2f2;
  border-radius: var(--overview-card-radius);
  border: 1px solid transparent;
  padding: 10px;
  box-shadow: none;
  cursor: pointer;
  transition:
    border-color 0.2s ease,
    background-color 0.2s ease;
  position: relative;
  overflow: hidden;
}

.score-card.score-card--in-compare {
  border: 1px solid var(--overview-card-border-in-compare);
  box-shadow: 0 2px 12px rgba(91, 126, 201, 0.12);
}

.score-card:hover {
  background: var(--overview-card-hover-bg);
  box-shadow: none;
  border-color: transparent;
}

/* 悬停时与示意图一致：标题加深、标签反色、对比钮深色底白图标 */
.score-card:hover .card-brand-name {
  color: var(--overview-hover-title);
}

.score-card:hover .card-head-tags :deep(.ant-tag) {
  color: #fff !important;
  background: var(--overview-hover-tag-bg) !important;
  border: none !important;
}

.score-card:hover .card-head-tags :deep(.overview-tag--theme-blue.ant-tag) {
  color: #fff !important;
  background: var(--overview-hover-tag-bg) !important;
  border: none !important;
}

.score-card:hover .card-head-tags :deep(.overview-tag--theme-blue .adv-txt-num) {
  color: #fff !important;
}

.score-card::before {
  display: none;
}

.card-head {
  --card-logo-size: 58px;
  --card-head-inline-gap: 12px;
  /* 与 tag 同高，+/- 按钮宽高与之对齐 */
  --overview-chip-h: 28px;
  margin-bottom: 16px;
}

.card-head-row--main {
  display: flex;
  align-items: center;
  gap: var(--card-head-inline-gap);
  min-width: 0;
}

.card-head-center {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px 15px;
}

.card-brand-name {
  font-size: 18px;
  font-weight: 900;
  color: var(--overview-brand-name);
  letter-spacing: 0.02em;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  transition: color 0.2s ease;
}

.card-head-tags {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  min-width: 0;
}

.card-head-tags :deep(.ant-tag) {
  margin: 0 !important;
  border-radius: 4px !important;
  font-size: 14px !important;
  line-height: 1.35 !important;
  min-height: var(--overview-chip-h) !important;
  box-sizing: border-box !important;
  display: inline-flex !important;
  align-items: center !important;
  padding: 2px 10px !important;
  color: var(--overview-muted) !important;
  background: #fff !important;
  border: none !important;
  transition:
    background-color 0.2s ease,
    color 0.2s ease;
}

.card-head-tags :deep(.overview-tag--theme-blue.ant-tag) {
  color: var(--overview-muted) !important;
  background: #fff !important;
  border: none !important;
}

.card-head-tags :deep(.overview-tag--theme-blue .adv-txt-num) {
  color: var(--overview-muted) !important;
}

.adv-txt-num {
  font-weight: 700;
}

.card-head-compare-btn {
  flex-shrink: 0;
}

/* 对比：与 tag 同尺寸方块、无边框，仅图标（文案保留在 DOM 供读屏） */
.card-head-compare-btn.ant-btn.ant-btn-text {
  border-radius: 4px;
  border: none !important;
  box-shadow: none;
  font-size: 14px;
  color: var(--overview-muted);
  background: #fff;
  padding: 0 !important;
  width: var(--overview-chip-h) !important;
  height: var(--overview-chip-h) !important;
  min-width: var(--overview-chip-h) !important;
  min-height: var(--overview-chip-h) !important;
  line-height: 1 !important;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  transition:
    color 0.2s ease,
    background-color 0.2s ease;
}

/* 卡片悬停或按钮自身悬停/聚焦：深色方钮 + 白图标（替代原先主题蓝描边效果） */
.score-card:hover .card-head-compare-btn.ant-btn:not(.card-head-compare-btn--in),
.card-head-compare-btn.ant-btn:not(.card-head-compare-btn--in):hover,
.card-head-compare-btn.ant-btn:not(.card-head-compare-btn--in):focus-visible {
  color: #fff;
  background: var(--overview-hover-btn-bg);
  border: none !important;
  box-shadow: none;
}

/* 仅当卡片未悬停时收敛「鼠标焦点但非 :focus-visible」的底色，避免盖住 .score-card:hover 下的深色按钮 */
.score-card:not(:hover) .card-head-compare-btn.ant-btn:not(.card-head-compare-btn--in):focus:not(:focus-visible):not(:hover) {
  color: var(--overview-muted);
  background: #fff;
  border: none !important;
  box-shadow: none;
}

.card-head-compare-btn__inner {
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

.card-head-compare-btn__txt {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

.card-head-compare-btn__ico {
  font-size: 14px;
  color: inherit;
}

.card-head-compare-btn--in.ant-btn.ant-btn-text {
  color: var(--purple);
  background: #fff !important;
  border: none !important;
  box-shadow: none;
}

.score-card:hover .card-head-compare-btn--in.ant-btn.ant-btn-text {
  color: #fff !important;
  background: var(--overview-hover-btn-in-bg) !important;
  border: none !important;
}

.card-head-compare-btn--in.ant-btn.ant-btn-text:hover,
.card-head-compare-btn--in.ant-btn.ant-btn-text:focus-visible {
  color: var(--purple);
  background: #faf8ff !important;
  border: none !important;
}

.score-card:hover .card-head-compare-btn--in.ant-btn.ant-btn-text:hover,
.score-card:hover .card-head-compare-btn--in.ant-btn.ant-btn-text:focus-visible {
  color: #fff !important;
  background: color-mix(in srgb, var(--overview-hover-btn-in-bg) 92%, #000) !important;
  border: none !important;
}

.card-head-compare-btn--in.ant-btn.ant-btn-text:focus:not(:focus-visible):not(:hover) {
  color: var(--purple);
  background: #fff !important;
  border: none !important;
}

.score-card:hover .card-head-compare-btn--in.ant-btn.ant-btn-text:focus:not(:focus-visible):not(:hover) {
  color: #fff !important;
  background: var(--overview-hover-btn-in-bg) !important;
  border: none !important;
}

.card-logo {
  width: var(--card-logo-size);
  height: var(--card-logo-size);
  flex-shrink: 0;
  padding: 4px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  background: #fff;
  border: none;
  box-sizing: border-box;
}

.card-logo-img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  padding: 0;
  display: block;
}

.card-logo-fallback {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #fff;
  font-size: 14px;
  font-weight: 700;
  border-radius: 8px;
}

.card-scores {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  margin-bottom: 18px;
}

.card-scores--single {
  grid-template-columns: 1fr;
}

.score-col {
  text-align: center;
  padding: 5px 12px;
  border-radius: 8px;
  background: #fff;
  border: none;
  box-sizing: border-box;
}

.sc-label {
  font-size: 14px;
  font-weight: 600;
}

.sc-num {
  font-size: 32px;
  font-weight: 700;
  line-height: 1.05;
  letter-spacing: -0.02em;
  margin-bottom: 2px;
}

.sc-cap {
  font-size: 14px;
  line-height: 1.25;
  margin-bottom: 6px;
}

.sc-cap--base {
  color: var(--overview-muted);
}

.sc-label.sc-label--base {
  color: var(--overview-muted);
}

.sc-num.sc-num--base {
  color: var(--overview-muted);
}

.sc-val.sc-val--base {
  color: var(--overview-muted);
  font-weight: 600;
}

.sc-val {
  font-size: 14px;
  font-weight: 600;
}

.stat-lbl {
  font-size: 14px;
  color: var(--overview-caption);
  margin-bottom: 4px;
  white-space: nowrap;
}

.stat-val {
  font-size: 16px;
  font-weight: 600;
  color: var(--overview-muted);
}

.stat-val--wrap {
  white-space: normal;
  overflow-wrap: anywhere;
  word-break: break-word;
}

.card-stats {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 0;
  margin: 0;
  background: transparent;
  border: none;
  box-shadow: none;
}

.stat-item {
  flex: 1 1 0;
  min-width: 0;
  text-align: center;
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
