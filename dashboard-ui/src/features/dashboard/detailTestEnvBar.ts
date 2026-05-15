/**
 * 详情页「测试环境」横条：n_gpu · date · device（无则省略前三项中有缺失的项；device 按维度规则必填展示）。
 */

import { formatDisplayDateYmd } from '@/utils/formatDisplayDate'
import type { BwDetailRow } from '@/features/dashboard/bwBenchmark'
import { pickBestBwRow } from '@/features/dashboard/bwBenchmark'

/** 算子详情：固定硬件型号（按平台） */
export const DETAIL_TEST_ENV_OP_DEVICE: Record<string, string> = {
  cambricon: '寒武纪 MLU590-M9C',
  hygon: '海光 BW C-3000',
  mthreads: '摩尔 MTT S5000',
  metax: '沐曦 C550',
  nvidia: 'NVIDIA',
  ascend: '昇腾 910B4',
  iluvatar: '天数 TG-V200',
  generic: '阿里 PPU',
}

/** 训练详情：固定硬件型号 */
export const DETAIL_TEST_ENV_TRAIN_DEVICE: Record<string, string> = {
  nvidia: '英伟达 A800',
  metax: '沐曦 C550',
  ascend: '昇腾 910B4',
  mthreads: '摩尔 S5000',
}

/** 通信详情：固定硬件型号 */
export const DETAIL_TEST_ENV_COMM_DEVICE: Record<string, string> = {
  nvidia: 'NVIDIA',
  metax: '沐曦 C550',
  cambricon: '寒武纪 MLU590',
}

function joinParts(parts: Array<string | undefined | null | false>): string {
  return parts.filter((x) => x != null && String(x).trim() !== '').join(' · ')
}

/** 横条内日期展示：YYYY-MM-DD；无法识别时保留原文 */
function barDateDisplay(raw: string): string {
  const s = String(raw).trim()
  if (!s) return ''
  const ymd = formatDisplayDateYmd(s)
  return ymd || s
}

/** 多行里的 date 去重后取字典序最后（与算子/训练 CSV 常见格式一致） */
function pickBarDateFromRows(rows: { date?: string }[]): string {
  const dates = [...new Set(rows.map((x) => String(x.date ?? '').trim()).filter(Boolean))]
  if (dates.length === 1) return dates[0]!
  if (dates.length > 1) return dates.sort().slice(-1)[0]!
  return ''
}

function ngpuPart(n: number | undefined): string | undefined {
  if (n == null || !Number.isFinite(n) || n <= 0) return undefined
  return `${Math.round(n)} GPU`
}

/** 访存：CSV model → 展示用硬件文案 */
export function bwModelsLineForPlatform(platKey: string, models: string[]): string {
  const uniq = [...new Set(models.map((m) => String(m || '').trim()).filter(Boolean))]
  if (!uniq.length) return ''
  const labels = uniq.map((m) => {
    switch (platKey) {
      case 'nvidia':
        return `英伟达 ${m}`
      case 'cambricon':
        return `寒武纪 ${m}`
      case 'metax':
        return `沐曦 ${m}`
      case 'mthreads':
        return `摩尔 ${m}`
      case 'ascend':
        return `昇腾 ${m}`
      case 'iluvatar':
        return `天数 ${m}`
      case 'hygon':
        return `海光 ${m}`
      default:
        return m
    }
  })
  return [...new Set(labels)].join(' / ')
}

export type InferTestEnvRow = {
  tps: number
  nGpu?: number
  remarks?: string
  date?: string
}

/**
 * 推理横条：优先当前 Batch/In-len 与 Prefill/Decode Tab 下的行；若无行或胜出行缺 date，
 * 再在同平台 prefill+decode 全量行上回退日期（与同平台推理 CSV 一致）。
 */
export function buildInferTestEnvLine(
  rows: InferTestEnvRow[],
  platformWideRows?: InferTestEnvRow[],
): string {
  const primary = rows.length ? rows : (platformWideRows ?? [])
  if (!primary.length) return '暂无推理数据'
  const r = primary.reduce((a, b) => (a.tps >= b.tps ? a : b))
  let dateStr = String(r.date ?? '').trim()
  if (!dateStr) dateStr = pickBarDateFromRows(primary)
  if (!dateStr && platformWideRows?.length && rows.length)
    dateStr = pickBarDateFromRows(platformWideRows)
  const device = String(r.remarks ?? '').trim()
  return joinParts([
    ngpuPart(r.nGpu),
    dateStr ? barDateDisplay(dateStr) : undefined,
    device || undefined,
  ])
}

export type OpTestEnvRow = { date?: string }

/**
 * 算子横条日期：优先当前筛选下的明细行；若该算子类型在导入数据里不存在（例如 CSV 未含该 op），
 * 则回退到本平台 **全部算子类型** 的行上的 date，与同一份 operator CSV 的测试批次一致。
 */
export function buildOpTestEnvLine(
  platKey: string,
  rows: OpTestEnvRow[],
  platformWideRows?: OpTestEnvRow[],
): string {
  const device =
    DETAIL_TEST_ENV_OP_DEVICE[platKey] ?? DETAIL_TEST_ENV_OP_DEVICE.generic ?? '—'
  let dateStr = pickBarDateFromRows(rows)
  if (!dateStr && platformWideRows?.length) dateStr = pickBarDateFromRows(platformWideRows)
  return joinParts([dateStr ? barDateDisplay(dateStr) : undefined, device])
}

export type TrainTestEnvRow = { tps: number; nGpu?: number; date?: string }

/**
 * 训练横条：优先当前「框架」筛选下的行；无行则退回全平台训练行。
 * 胜出行若无 date，则在当前集合 / 全平台集合上回退任一带 date 的行（与同平台训练表一致）。
 */
export function buildTrainTestEnvLine(
  platKey: string,
  rows: TrainTestEnvRow[],
  platformWideRows?: TrainTestEnvRow[],
  /** `*_train_YYYYMMDD.xlsx` 文件名解析，行内无 date 时使用 */
  sourceFileDate?: string,
): string {
  const primary = rows.length ? rows : (platformWideRows ?? [])
  if (!primary.length) return '暂无训练数据'
  const r = primary.reduce((a, b) => (a.tps >= b.tps ? a : b))
  const device =
    DETAIL_TEST_ENV_TRAIN_DEVICE[platKey] ??
    DETAIL_TEST_ENV_OP_DEVICE[platKey] ??
    '—'
  let dateStr = String(r.date ?? '').trim()
  if (!dateStr) dateStr = pickBarDateFromRows(primary)
  if (!dateStr && platformWideRows?.length && rows.length) dateStr = pickBarDateFromRows(platformWideRows)
  if (!dateStr) dateStr = String(sourceFileDate ?? '').trim()
  return joinParts([
    ngpuPart(r.nGpu),
    dateStr ? barDateDisplay(dateStr) : undefined,
    device,
  ])
}

export type CommTestEnvRow = { bw: number; nGpu?: number; date?: string }

/**
 * 通信横条：优先当前「通信类型」筛选；无行则退回全表。
 * 日期优先代表行（bw 最高）的 `date`；若为空则在当前集合 / 全表行上取行内 `date` 字典序最大（与侧栏、文件 `date` 列一致）。
 */
export function buildCommTestEnvLine(
  platKey: string,
  rows: CommTestEnvRow[],
  platformWideRows?: CommTestEnvRow[],
  /** 保留参数供调用方兼容；横条日期不以文件名为准 */
  _sourceFileDate?: string,
): string {
  const primary = rows.length ? rows : (platformWideRows ?? [])
  if (!primary.length) return '暂无通信数据'
  const r = primary.reduce((a, b) => (a.bw >= b.bw ? a : b))
  const device =
    DETAIL_TEST_ENV_COMM_DEVICE[platKey] ??
    DETAIL_TEST_ENV_OP_DEVICE[platKey] ??
    '—'
  let dateStr = String(r.date ?? '').trim()
  if (!dateStr) dateStr = pickBarDateFromRows(primary)
  if (!dateStr && platformWideRows?.length && rows.length) dateStr = pickBarDateFromRows(platformWideRows)
  return joinParts([ngpuPart(r.nGpu), dateStr ? barDateDisplay(dateStr) : undefined, device])
}

export function buildBwTestEnvLine(
  platKey: string,
  rows: BwDetailRow[],
  /** 保留参数供调用方兼容；日期优先代表行，缺省则取行内 date 最大（与其它维度一致） */
  _sourceFileDate?: string,
): string {
  if (!rows.length) return '暂无访存数据'
  const models = rows.map((x) => x.model)
  const device = bwModelsLineForPlatform(platKey, models)
  const best = pickBestBwRow(rows) ?? rows[0]
  let dateStr = String(best?.date ?? '').trim()
  if (!dateStr) dateStr = pickBarDateFromRows(rows)
  const remarks = String(best?.remarks ?? '').trim()
  const tester = String(best?.tester ?? '').trim()
  return joinParts([
    dateStr ? barDateDisplay(dateStr) : undefined,
    device || undefined,
    remarks || undefined,
    tester || undefined,
  ])
}

/** 详情页「测试环境」横条下方来源说明（固定文案） */
export const DETAIL_TEST_ENV_SOURCE_HINT = '来源：JSON 测试输出，接入后自动填充。'
