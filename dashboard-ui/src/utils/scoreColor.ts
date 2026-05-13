/**
 * 全站统一「得分 / vs NVIDIA」色阶工具。
 *
 * 产品规则（概览卡 & 详情五大维度均一致）：
 *   - 得分 ≥ 100  → 绿色（达到或超过 NVIDIA 基线）— 用于数字/柱图等前景色
 *   - 60 ≤ 得分 < 100 → 橙色（接近基线，但仍有差距）
 *   - 得分 < 60   → 红色（差距较大）
 *   - 缺失 / 不可比 → 灰色
 *
 * 概览卡 Ant Tag（国产 / 自研快 advTxt）高分档使用蓝色预设，与「国际标杆」主题一致，
 * 见 `SCORE_TIER_TAG_PRESET`；其它展示仍用 `scoreTierColor`。
 *
 * 任何展示 vs NVIDIA 百分比、或基于该百分比的得分（含算子综合得分）的地方，
 * 都应直接使用本文件提供的 helper，禁止再就地写 `>= 100 ? 绿 : 红` 之类的二档判断。
 */

export type ScoreTier = 'high' | 'mid' | 'low' | 'none'

/** 与 Ant Design 色板对齐的统一色值 */
export const SCORE_TIER_COLOR: Record<ScoreTier, string> = {
  high: '#2e7d32',
  mid: '#e65100',
  low: '#c62828',
  none: '#8c8c8c',
}

/** Ant Tag 预设（概览卡 advTxt 等）：高分档用蓝，与「国际标杆」及全站 --blue 一致 */
export const SCORE_TIER_TAG_PRESET: Record<ScoreTier, string> = {
  high: 'blue',
  mid: 'orange',
  low: 'red',
  none: 'default',
}

/** 将得分（百分比，可空）映射到色阶档位 */
export function scoreTier(score: number | null | undefined): ScoreTier {
  if (score == null || !Number.isFinite(score)) return 'none'
  if (score >= 100) return 'high'
  if (score >= 60) return 'mid'
  return 'low'
}

/** 直接拿得分对应的前景色（最常用入口） */
export function scoreTierColor(score: number | null | undefined): string {
  return SCORE_TIER_COLOR[scoreTier(score)]
}

/** Ant Tag 预设：用于概览卡 advTxt 等场景 */
export function scoreTierTagPreset(score: number | null | undefined): string {
  return SCORE_TIER_TAG_PRESET[scoreTier(score)]
}

/**
 * ECharts 柱图填充色：在三档基础上统一附加 `cc`（≈80%）alpha，
 * 与详情区原有「vs NVIDIA」柱图视觉风格保持一致。
 * 缺失档（none）不使用透明度，便于上层自行决定是否渲染。
 */
export function scoreTierBarColor(score: number | null | undefined): string {
  const tier = scoreTier(score)
  if (tier === 'none') return SCORE_TIER_COLOR.none
  return SCORE_TIER_COLOR[tier] + 'cc'
}
