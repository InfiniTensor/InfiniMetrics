/**
 * 从仓库根目录 `new_data/operator`、`new_data/infer`、`new_data/train`、`new_data/comm`、`new_data/bw` 生成 `src/data/generatedFromFiles.ts`。
 *
 * 前端读的是本脚本生成进 `generatedFromFiles.ts` 的数据，不会在浏览器里直接打开 Excel/CSV。
 * 改 `new_data/**` 后必须执行：`npm run generate:data`（`npm run dev` / `npm run build` 已配置为执行前自动生成）。
 *
 * 通信：`new_data/comm/{plat}_comm_YYYYMMDD.xlsx` 或同名的 `.csv`（同平台同时存在时优先 CSV）；文件名 8 位日期会参与行内日期的对齐。列：link_type、comm_type、n_gpu、bw_GBps（表头规范化后为 bw_gbps）、date、remarks。
 * 访存：`{plat}_bw_YYYYMMDD.csv` 或 `.xlsx`（同平台同时存在时优先 CSV）、可选的 `bw/bw_template.csv`（多平台、含 platform 列）。列：model、add_bw_GBps、copy_bw_GBps、scale_bw_GBps、triad_bw_GBps、bw_GBps、date、tester、remarks（表头规范化后为下划线小写）。
 * 算子：`{plat}_operator_YYYYMMDD.csv` 或 `.xlsx`（同平台同时存在时优先日期较新者；日期相同优先 CSV）。推理 / 训练 / 通信规则见各 `*Benchmark.ts`。
 */
import { parse } from 'csv-parse/sync'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'
import * as XLSXImport from 'xlsx'
/** tsx / ESM 下 xlsx 常为 `default` 导出，`import *` 无 `readFile` */
const XLSX =
  typeof (XLSXImport as unknown as { readFile?: unknown }).readFile === 'function'
    ? (XLSXImport as unknown as typeof import('xlsx'))
    : ((XLSXImport as unknown as { default: typeof import('xlsx') }).default as typeof import('xlsx'))
import {
  buildInferCardMetrics,
  enrichInferWithNvidiaBaseline,
  inferConfigKey,
  INFER_FRAMEWORK,
  normalizeInferModel,
  type InferDecodeRow,
  type InferFlatRow,
  type InferPrefillRow,
} from '../src/features/dashboard/inferBenchmark'
import {
  buildOpPlatformOverview,
  canComputeOpRowScore,
  type OperatorTableRow,
} from '../src/features/dashboard/operatorBenchmark'
import {
  buildCommCardMetrics,
  commMatchKey,
  enrichCommBaselines,
  formatCommLinkType,
  normalizeCommType,
  type CommImportRow,
} from '../src/features/dashboard/commBenchmark'
import {
  buildTrainCardMetrics,
  buildTrainNote,
  formatTrainParallel,
  normalizeTrainDtype,
  parseFlashAttnCell,
  trainMatchKey,
  trainNvidiaJoinKey,
  trainVsPercent,
  type TrainTableRow,
} from '../src/features/dashboard/trainBenchmark'
import {
  buildBwCardMetrics,
  bwVsNvidiaPercent,
  type BwDetailRow,
} from '../src/features/dashboard/bwBenchmark'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const ROOT = path.join(__dirname, '..', '..')
const DATA_OPERATOR = path.join(ROOT, 'new_data', 'operator')
const DATA_INFER = path.join(ROOT, 'new_data', 'infer')
const DATA_TRAIN = path.join(ROOT, 'new_data', 'train')
const DATA_COMM = path.join(ROOT, 'new_data', 'comm')
const DATA_BW = path.join(ROOT, 'new_data', 'bw')
const OUT_FILE = path.join(__dirname, '..', 'src', 'data', 'generatedFromFiles.ts')

/** CSV 中的平台前缀 → 仪表盘 PLATFORMS.key */
const PLATFORM_ALIAS: Record<string, string> = {
  moore: 'mthreads',
  ali: 'generic',
  nvdia: 'nvidia',
  nivdia: 'nvidia',
}

function normalizePlatform(raw: string | undefined): string {
  const k = String(raw || '').trim().toLowerCase()
  return PLATFORM_ALIAS[k] ?? k
}

function parseNum(v: unknown): number {
  if (v == null || v === '') return NaN
  const s = String(v).trim().toUpperCase()
  if (s === 'NA' || s === 'N/A') return NaN
  const n = Number.parseFloat(String(v))
  return Number.isFinite(n) ? n : NaN
}

/** op_name → 与 dashboard DIMS pills / OP_TABLE 键一致 */
function opNameToKey(name: string | undefined): string {
  const lower = String(name || '').trim().toLowerCase().replace(/\s+/g, '')
  const map: Record<string, string> = {
    causalsoftmax: 'CausalSoftmax',
    rmsnorm: 'RMSNorm',
    embedding: 'Embedding',
    topk: 'TopK',
    matmul: 'MatMul',
    add: 'Add',
    silu: 'SiLU',
    cast: 'Cast',
    cat: 'Cat',
  }
  if (map[lower]) return map[lower]
  return lower ? lower.charAt(0).toUpperCase() + lower.slice(1) : 'Unknown'
}

function normalizeShape(s: unknown): string {
  return String(s || '')
    .trim()
    .replace(/^"|"$/g, '')
    .replace(/,\s*/g, ', ')
}

function parseOperatorRecords(
  records: Record<string, string>[],
  forcedPlatform: string | undefined,
): { byPlat: Record<string, Record<string, OperatorTableRow[]>>; flatByPlat: Record<string, OperatorFlat[]> } {
  const byPlat: Record<string, Record<string, OperatorTableRow[]>> = {}
  const flatByPlat: Record<string, OperatorFlat[]> = {}

  for (const row of records) {
    const plat = forcedPlatform ?? normalizePlatform(row.platform)
    if (!plat) continue
    const opKey = opNameToKey(row.op_name)
    const ic = parseNum(row.ic_latency_ms)
    const pt = parseNum(row.pt_latency_ms)
    const remarks = String(row.remarks ?? '').trim()
    const dtype = String(row.dtype || '').trim().toUpperCase()
    const shape = normalizeShape(row.shape_config)
    const date = String(row.date || '').trim()

    if (!Number.isFinite(ic) || !Number.isFinite(pt)) continue

    const scoreEligible = canComputeOpRowScore(ic, pt, remarks)
    const outRow: OperatorTableRow = {
      shape,
      dtype,
      ic,
      pt,
      remarks,
      scoreEligible,
    }
    if (date) outRow.date = date

    if (!byPlat[plat]) byPlat[plat] = {}
    if (!byPlat[plat][opKey]) byPlat[plat][opKey] = []
    byPlat[plat][opKey].push(outRow)

    if (!flatByPlat[plat]) flatByPlat[plat] = []
    flatByPlat[plat].push({ opKey, shape, dtype, ic, pt, remarks, date })
  }
  return { byPlat, flatByPlat }
}

function parseOperatorCsv(
  filePath: string,
  forcedPlatform: string | undefined,
): { byPlat: Record<string, Record<string, OperatorTableRow[]>>; flatByPlat: Record<string, OperatorFlat[]> } {
  const text = fs.readFileSync(filePath, 'utf8').replace(/^\uFEFF/, '')
  const records = parse(text, {
    columns: true,
    skip_empty_lines: true,
    trim: true,
    relax_column_count: true,
  }) as Record<string, string>[]
  return parseOperatorRecords(records, forcedPlatform)
}

function parseOperatorFromFile(
  filePath: string,
  forcedPlatform: string | undefined,
): { byPlat: Record<string, Record<string, OperatorTableRow[]>>; flatByPlat: Record<string, OperatorFlat[]> } {
  if (filePath.toLowerCase().endsWith('.xlsx')) {
    return parseOperatorRecords(readFirstSheetStringRecords(filePath), forcedPlatform)
  }
  return parseOperatorCsv(filePath, forcedPlatform)
}

/** 与访存一致：各平台取最新日期的 csv 或 xlsx；同日优先 csv */
function pickLatestOperatorByPlatform(): Record<string, { date: string; full: string }> {
  const csvMap = pickLatestByPlatform(DATA_OPERATOR, /^(.+)_operator_(\d+)\.csv$/i)
  const xlsxMap = pickLatestByPlatform(DATA_OPERATOR, /^(.+)_operator_(\d+)\.xlsx$/i)
  const plats = new Set([...Object.keys(csvMap), ...Object.keys(xlsxMap)])
  const out: Record<string, { date: string; full: string }> = {}
  for (const plat of plats) {
    const c = csvMap[plat]
    const x = xlsxMap[plat]
    if (!c) out[plat] = x!
    else if (!x) out[plat] = c
    else if (c.date !== x.date) out[plat] = c.date > x.date ? c : x
    else out[plat] = c
  }
  return out
}

type OperatorFlat = {
  opKey: string
  shape: string
  dtype: string
  ic: number
  pt: number
  remarks: string
  date: string
}

function pickPrefillDecodeKeys(sample: Record<string, string>): { pk: string; dk: string } | null {
  const keys = Object.keys(sample)
  const pk =
    keys.find((k) => {
      const kl = k.toLowerCase()
      return /prefill/i.test(kl) && (/token/i.test(kl) || /tokens/i.test(kl) || /吞吐/.test(kl))
    }) ??
    keys.find((k) => /prefill吞吐/i.test(k))
  const dk =
    keys.find((k) => {
      const kl = k.toLowerCase()
      return (
        /^decode/i.test(kl) &&
        (/token/i.test(kl) || /tokens/i.test(kl) || /吞吐/.test(kl)) &&
        !/il_decode|decode_ms|vl_decode/i.test(kl)
      )
    }) ?? keys.find((k) => /decode吞吐/i.test(k))
  if (pk && dk) return { pk, dk }
  return null
}

/** `new_data/.../nvidia_train_20260506.xlsx` 文件名中的 8 位日期 → `2026-05-06` */
function fileStampYmdToIso(stamp: string): string {
  const s = String(stamp || '').trim()
  if (/^\d{8}$/.test(s)) return `${s.slice(0, 4)}-${s.slice(4, 6)}-${s.slice(6, 8)}`
  return ''
}

function inferDateToSortable(s: string): string {
  const t = String(s || '').trim()
  const mYmd = t.match(/^(\d{4})\/(\d{1,2})\/(\d{1,2})$/)
  if (mYmd) {
    return `${mYmd[1]}-${mYmd[2].padStart(2, '0')}-${mYmd[3].padStart(2, '0')}`
  }
  const mUs = t.match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})$/)
  if (mUs) {
    const mo = mUs[1].padStart(2, '0')
    const da = mUs[2].padStart(2, '0')
    return `${mUs[3]}-${mo}-${da}`
  }
  const mIso = t.match(/^(\d{4})-(\d{2})-(\d{2})/)
  if (mIso) return `${mIso[1]}-${mIso[2]}-${mIso[3]}`
  const mCn = t.match(/^(\d{4})\/(\d{2})\/(\d{2})/)
  if (mCn) return `${mCn[1]}-${mCn[2]}-${mCn[3]}`
  return t
}

function parseInferCsvNew(
  filePath: string,
  forcedPlatform: string | undefined,
): { flats: InferFlatRow[]; prefill: InferPrefillRow[]; decode: InferDecodeRow[] } {
  const text = fs.readFileSync(filePath, 'utf8').replace(/^\uFEFF/, '')
  const records = parse(text, {
    columns: true,
    skip_empty_lines: true,
    trim: true,
    relax_column_count: true,
  }) as Record<string, string>[]
  if (!records.length) return { flats: [], prefill: [], decode: [] }

  const keysPick = pickPrefillDecodeKeys(records[0])
  if (!keysPick) {
    console.warn('infer CSV: cannot detect Prefill/Decode throughput columns', filePath)
    return { flats: [], prefill: [], decode: [] }
  }
  const { pk: prefillCol, dk: decodeCol } = keysPick

  const flats: InferFlatRow[] = []

  for (const row of records) {
    const plat = forcedPlatform ?? normalizePlatform(row.platform)
    if (!plat) continue
    const batch = Number.parseInt(String(row.batch_size ?? '').trim(), 10)
    const inLen = Number.parseInt(String(row.input_tokens ?? '').trim(), 10)
    const outLen = Number.parseInt(String(row.output_tokens ?? '').trim(), 10)
    if (!Number.isFinite(batch) || !Number.isFinite(inLen) || !Number.isFinite(outLen)) continue

    const prefillTps = parseNum(row[prefillCol])
    const decodeTps = parseNum(row[decodeCol])
    const ttftMs = parseNum(row.il_ttft_ms)
    const decodeMs = parseNum(row.il_decode_ms)

    if (!Number.isFinite(prefillTps) && !Number.isFinite(decodeTps)) continue

    flats.push({
      plat,
      batch,
      inLen,
      outLen,
      model: normalizeInferModel(row.model),
      dtype: String(row.dtype || '').trim().toUpperCase(),
      nGpu: Number.parseInt(String(row.n_gpu ?? '').trim(), 10) || 0,
      remarks: String(row.remarks ?? '').trim(),
      date: String(row.date ?? '').trim(),
      ttftMs: Number.isFinite(ttftMs) ? ttftMs : null,
      decodeMs: Number.isFinite(decodeMs) ? decodeMs : null,
      prefillTps: Number.isFinite(prefillTps) ? prefillTps : null,
      decodeTps: Number.isFinite(decodeTps) ? decodeTps : null,
    })
  }

  flats.sort((a, b) => {
    if (a.batch !== b.batch) return a.batch - b.batch
    if (a.inLen !== b.inLen) return a.inLen - b.inLen
    return a.outLen - b.outLen
  })

  const prefill: InferPrefillRow[] = []
  const decode: InferDecodeRow[] = []

  for (const r of flats) {
    const key = inferConfigKey(r.batch, r.inLen, r.outLen)
    if (r.prefillTps != null) {
      const preRow: InferPrefillRow = {
        configKey: key,
        batch: r.batch,
        inLen: r.inLen,
        outLen: r.outLen,
        model: r.model,
        dtype: r.dtype,
        nGpu: r.nGpu,
        remarks: r.remarks,
        tps: Math.round(r.prefillTps),
        ttft: r.ttftMs != null && r.ttftMs > 0 ? r.ttftMs : undefined,
        decodeLatencyMs: r.decodeMs != null && r.decodeMs > 0 ? r.decodeMs : undefined,
        vsNvidia: null,
        nvidiaBaselineTps: null,
        framework: INFER_FRAMEWORK,
      }
      if (r.date) preRow.date = r.date
      prefill.push(preRow)
    }
    if (r.decodeTps != null) {
      const decRow: InferDecodeRow = {
        configKey: key,
        batch: r.batch,
        inLen: r.inLen,
        outLen: r.outLen,
        model: r.model,
        dtype: r.dtype,
        nGpu: r.nGpu,
        remarks: r.remarks,
        tps: Math.round(r.decodeTps),
        decodeLatencyMs: r.decodeMs != null && r.decodeMs > 0 ? r.decodeMs : undefined,
        vsNvidia: null,
        nvidiaBaselineTps: null,
        framework: INFER_FRAMEWORK,
      }
      if (r.date) decRow.date = r.date
      decode.push(decRow)
    }
  }

  return { flats, prefill, decode }
}

function pickLatestByPlatform(dir: string, pattern: RegExp) {
  const map: Record<string, { date: string; full: string }> = {}
  if (!fs.existsSync(dir)) return map
  for (const name of fs.readdirSync(dir)) {
    const m = name.match(pattern)
    if (!m) continue
    const plat = normalizePlatform(m[1])
    const date = m[2]
    const full = path.join(dir, name)
    if (!map[plat] || date > map[plat].date) {
      map[plat] = { date, full }
    }
  }
  return map
}

function normalizeXlsxHeaderKey(k: string): string {
  return String(k || '')
    .trim()
    .toLowerCase()
    .replace(/\u00a0/g, ' ')
    .replace(/\s+/g, '_')
}

/** 单元格 → 导入用字符串：`raw:false` 时多为已格式化文本；数字则可能是 Excel 日期序列号 */
function spreadsheetCellToImportString(v: unknown): string {
  if (v == null || v === '') return ''
  if (typeof v === 'number' && Number.isFinite(v)) {
    const n = v
    if (n > 20000 && n < 120000) {
      const whole = Math.floor(n)
      const ms = (whole - 25569) * 86400 * 1000
      const d = new Date(ms)
      return `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, '0')}-${String(d.getUTCDate()).padStart(2, '0')}`
    }
  }
  if (v instanceof Date) {
    return `${v.getFullYear()}-${String(v.getMonth() + 1).padStart(2, '0')}-${String(v.getDate()).padStart(2, '0')}`
  }
  return String(v).trim()
}

function readFirstSheetStringRecords(filePath: string): Record<string, string>[] {
  if (!fs.existsSync(filePath)) return []
  const wb = XLSX.readFile(filePath)
  const sheetName = wb.SheetNames[0]
  if (!sheetName) return []
  const sheet = wb.Sheets[sheetName]
  const raw = XLSX.utils.sheet_to_json<Record<string, unknown>>(sheet, {
    defval: '',
    raw: false,
  })
  return raw.map((rec) => {
    const o: Record<string, string> = {}
    for (const [k, v] of Object.entries(rec)) {
      o[normalizeXlsxHeaderKey(k)] = spreadsheetCellToImportString(v)
    }
    return o
  })
}

/** 解析出的日历日与文件名 `*_YYYYMMDD` 一致时，用文件名年月日（解决 Excel `4/28/25` 与 `20260428` 年份不一致） */
function alignIsoDateToFileHint(parsedIso: string, fileHintIso?: string): string {
  if (!fileHintIso || !/^\d{4}-\d{2}-\d{2}$/.test(fileHintIso)) return parsedIso
  if (!/^\d{4}-\d{2}-\d{2}$/.test(parsedIso)) return parsedIso
  if (parsedIso.slice(5) === fileHintIso.slice(5)) return fileHintIso
  return parsedIso
}

/** 通信/训练等：ISO、年在前斜杠、美区 M/D/YY、纯序列号字符串 → YYYY-MM-DD */
function normalizeImportedCalendarDateString(s: string, fileHintIso?: string): string {
  const t = String(s ?? '').trim()
  if (!t) return ''
  const iso = t.match(/^(\d{4}-\d{2}-\d{2})/)
  if (iso) return alignIsoDateToFileHint(iso[1]!, fileHintIso)
  const slash = t.match(/^(\d{4})[/.](\d{1,2})[/.](\d{1,2})/)
  if (slash) {
    const y = slash[1]
    const mo = slash[2].padStart(2, '0')
    const da = slash[3].padStart(2, '0')
    return alignIsoDateToFileHint(`${y}-${mo}-${da}`, fileHintIso)
  }
  const mdy4 = t.match(/^(\d{1,2})[/.](\d{1,2})[/.](\d{4})$/)
  if (mdy4) {
    const mo = mdy4[1].padStart(2, '0')
    const da = mdy4[2].padStart(2, '0')
    const y = mdy4[3]
    return alignIsoDateToFileHint(`${y}-${mo}-${da}`, fileHintIso)
  }
  const mdy2 = t.match(/^(\d{1,2})[/.](\d{1,2})[/.](\d{2})$/)
  if (mdy2) {
    let y = parseInt(mdy2[3], 10)
    if (y < 100) y = y >= 70 ? 1900 + y : 2000 + y
    const mo = mdy2[1].padStart(2, '0')
    const da = mdy2[2].padStart(2, '0')
    return alignIsoDateToFileHint(`${y}-${mo}-${da}`, fileHintIso)
  }
  if (/^\d+(\.\d+)?$/.test(t)) {
    const num = parseFloat(t)
    if (Number.isFinite(num) && num > 20000 && num < 120000) {
      const whole = Math.floor(num)
      const ms = (whole - 25569) * 86400 * 1000
      const d = new Date(ms)
      const out = `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, '0')}-${String(d.getUTCDate()).padStart(2, '0')}`
      return alignIsoDateToFileHint(out, fileHintIso)
    }
  }
  return t
}

function trainCell(o: Record<string, string>, ...keys: string[]): string {
  for (const k of keys) {
    const v = o[k]
    if (v !== undefined && v !== '') return v
  }
  return ''
}

function trainRowFromRecord(o: Record<string, string>): TrainTableRow | null {
  const framework = trainCell(o, 'framework').toLowerCase()
  const modelRaw = trainCell(o, 'model').trim()
  const modelKey = modelRaw.toLowerCase()
  const nGpu = Math.round(parseNum(trainCell(o, 'n_gpu', 'ngpu')))
  const seqLen = Math.round(parseNum(trainCell(o, 'seq_len', 'seqlen', 'seq_length')))
  const tps = parseNum(trainCell(o, 'throughput_tpps', 'throughput', 'tpps'))
  if (!framework || !modelKey || !Number.isFinite(nGpu) || !Number.isFinite(seqLen) || !Number.isFinite(tps))
    return null

  const mbsRaw = parseNum(trainCell(o, 'micro_batch_size', 'microbatchsize', 'mbs'))
  const dtype = normalizeTrainDtype(trainCell(o, 'dtype'))
  const flashRaw = parseFlashAttnCell(trainCell(o, 'flash_attn', 'flashattention'))
  const note = buildTrainNote(trainCell(o, 'zero_stage', 'zerostage'), trainCell(o, 'remarks', 'remark'))
  const dateRaw = trainCell(
    o,
    'date',
    'test_date',
    'run_date',
    'data_date',
    'experiment_date',
    '测试日期',
    '测试时间',
  )
  const date = normalizeImportedCalendarDateString(dateRaw)
  const mk = trainMatchKey(framework, modelKey, nGpu, seqLen, dtype)

  const row: TrainTableRow = {
    matchKey: mk,
    framework,
    model: modelRaw || modelKey,
    parallel: formatTrainParallel(nGpu, seqLen),
    dtype,
    flashAttn: flashRaw,
    tps: Math.round(tps),
    baseline: 0,
    vsA100: 100,
    note,
    nGpu,
    seqLen,
    microBatchSize: Number.isFinite(mbsRaw) ? Math.round(mbsRaw) : 0,
  }
  if (date) row.date = date
  return row
}

function ingestTrainFile(filePath: string, forcedPlat: string, acc: Record<string, TrainTableRow[]>) {
  const records = readFirstSheetStringRecords(filePath)
  for (const o of records) {
    const platCell = normalizePlatform(trainCell(o, 'platform', 'plat', 'vendor', 'chip'))
    if (platCell && platCell !== forcedPlat) continue
    const row = trainRowFromRecord(o)
    if (!row) continue
    if (!acc[forcedPlat]) acc[forcedPlat] = []
    acc[forcedPlat].push(row)
  }
}

/** 按 trainNvidiaJoinKey = (framework, model, n_gpu) 对齐 NVIDIA：同键取 NVIDIA 最大 tpps 为基线，行级 vs = throughput_tpps ÷ 基线 × 100 */
function enrichTrainBaselines(byPlat: Record<string, TrainTableRow[]>) {
  const nv = byPlat.nvidia || []
  const baselineMap = new Map<string, number>()
  for (const r of nv) {
    const k = trainNvidiaJoinKey(r.framework, r.model, r.nGpu)
    const prev = baselineMap.get(k) ?? 0
    baselineMap.set(k, Math.max(prev, r.tps))
  }
  for (const plat of Object.keys(byPlat)) {
    for (const row of byPlat[plat]) {
      const k = trainNvidiaJoinKey(row.framework, row.model, row.nGpu)
      const b = baselineMap.get(k)
      if (b != null && b > 0) {
        row.baseline = b
        row.vsA100 = trainVsPercent(row.tps, b)
      } else {
        row.baseline = row.tps
        row.vsA100 = 100
      }
    }
  }
}

function stripTrainMatchKey(rows: TrainTableRow[]): Omit<TrainTableRow, 'matchKey'>[] {
  return rows.map(({ matchKey: _mk, ...rest }) => rest)
}

function commCell(o: Record<string, string>, ...keys: string[]): string {
  for (const k of keys) {
    const v = o[k]
    if (v !== undefined && v !== '') return v
  }
  return ''
}

function commRowFromRecord(o: Record<string, string>, fileHintIso?: string): CommImportRow | null {
  const commType = normalizeCommType(commCell(o, 'comm_type', 'commtype', 'type'))
  if (commType !== 'p2p' && commType !== 'allreduce') return null
  const nGpu = Math.round(parseNum(commCell(o, 'n_gpu', 'ngpu')))
  /** 规范列名 `bw_GBps`；normalizeXlsxHeaderKey / CSV 列名统一为小写下划线后为 `bw_gbps` */
  const bw = parseNum(commCell(o, 'bw_gbps', 'bw', 'bandwidth_gbps', 'bandwidth'))
  if (!Number.isFinite(nGpu) || !Number.isFinite(bw)) return null
  const linkRaw = commCell(o, 'link_type', 'linktype', 'link')
  const linkType = formatCommLinkType(linkRaw)
  const note = commCell(o, 'remarks', 'remark', 'note')
  const dateRaw = commCell(
    o,
    'date',
    'test_date',
    'run_date',
    'data_date',
    'experiment_date',
    'timestamp',
    '测试日期',
    '测试时间',
  )
  const date = normalizeImportedCalendarDateString(dateRaw, fileHintIso)
  const mk = commMatchKey(commType, nGpu)
  const row: CommImportRow = {
    matchKey: mk,
    linkType,
    commType,
    nGpu,
    bw: Number(bw),
    baseline: 0,
    vsA100: 100,
    note,
  }
  if (date) row.date = date
  return row
}

function ingestCommFile(
  filePath: string,
  forcedPlat: string,
  acc: Record<string, CommImportRow[]>,
  fileHintIso?: string,
) {
  const records = readFirstSheetStringRecords(filePath)
  for (const o of records) {
    const row = commRowFromRecord(o, fileHintIso)
    if (!row) continue
    if (!acc[forcedPlat]) acc[forcedPlat] = []
    acc[forcedPlat].push(row)
  }
}

function readCommCsvStringRecords(filePath: string): Record<string, string>[] {
  if (!fs.existsSync(filePath)) return []
  const text = fs.readFileSync(filePath, 'utf8').replace(/^\uFEFF/, '')
  const records = parse(text, {
    columns: true,
    skip_empty_lines: true,
    trim: true,
    relax_column_count: true,
  }) as Record<string, unknown>[]
  return records.map((rec) => {
    const o: Record<string, string> = {}
    for (const [k, v] of Object.entries(rec)) {
      o[normalizeXlsxHeaderKey(String(k))] = String(v ?? '').trim()
    }
    return o
  })
}

function ingestCommCsvFile(
  filePath: string,
  forcedPlat: string,
  acc: Record<string, CommImportRow[]>,
  fileHintIso?: string,
) {
  for (const o of readCommCsvStringRecords(filePath)) {
    const row = commRowFromRecord(o, fileHintIso)
    if (!row) continue
    if (!acc[forcedPlat]) acc[forcedPlat] = []
    acc[forcedPlat].push(row)
  }
}

function stripCommMatchKey(rows: CommImportRow[]): Omit<CommImportRow, 'matchKey'>[] {
  return rows.map(({ matchKey: _mk, ...rest }) => rest)
}

function normalizeCsvRowKeys(rec: Record<string, unknown>): Record<string, string> {
  const o: Record<string, string> = {}
  for (const [k, v] of Object.entries(rec)) {
    o[String(k).trim().toLowerCase().replace(/\u00a0/g, ' ').replace(/\s+/g, '_')] = String(v ?? '').trim()
  }
  return o
}

function bwCsvCell(o: Record<string, string>, ...keys: string[]): string {
  for (const k of keys) {
    if (o[k] !== undefined && o[k] !== '') return o[k]
  }
  return ''
}

function appendBwDetailMeta(row: BwDetailRow, o: Record<string, string>) {
  const remarks = bwCsvCell(o, 'remarks', 'remark').trim()
  const tester = bwCsvCell(o, 'tester', '测试者', 'operator').trim()
  const dateRaw = bwCsvCell(
    o,
    'date',
    'test_date',
    'run_date',
    'data_date',
    'experiment_date',
    '测试日期',
    '测试时间',
  )
  const date = normalizeImportedCalendarDateString(dateRaw)
  if (remarks) row.remarks = remarks
  if (tester) row.tester = tester
  if (date) row.date = date
}

function bwRowFromCsvRecord(o: Record<string, string>): BwDetailRow | null {
  const model = bwCsvCell(o, 'model').trim()
  if (!model) return null

  /** 文件列 `*_bw_GBps`；规范化后为 `*_bw_gbps` */
  const add = parseNum(bwCsvCell(o, 'add_bw_gbps', 'add_bw_g', 'add'))
  const copy = parseNum(bwCsvCell(o, 'copy_bw_gbps', 'copy_bw_g', 'copy'))
  const scale = parseNum(bwCsvCell(o, 'scale_bw_gbps', 'scale_bw_g', 'scale'))
  const triad = parseNum(bwCsvCell(o, 'triad_bw_gbps', 'triad_bw_g', 'triad'))
  /** 文件列 `bw_GBps`（均值）；规范化后为 `bw_gbps` */
  const avgRaw = parseNum(bwCsvCell(o, 'bw_gbps', 'bw', 'hbm_mean_gbps', 'mean_bw_gbps'))

  const addN = Number.isFinite(add) ? add : null
  const copyN = Number.isFinite(copy) ? copy : null
  const scaleN = Number.isFinite(scale) ? scale : null
  const triadN = Number.isFinite(triad) ? triad : null

  let avg: number | null = null
  if (Number.isFinite(avgRaw)) avg = avgRaw
  else if (addN != null && copyN != null && scaleN != null && triadN != null) {
    avg = (addN + copyN + scaleN + triadN) / 4
  }

  const modesKnown = [addN, copyN, scaleN, triadN].filter((x) => x != null).length
  if (avg == null && modesKnown === 0) {
    const row: BwDetailRow = {
      model,
      add: addN,
      copy: copyN,
      scale: scaleN,
      triad: triadN,
      avg: null,
      vsNvidia: 100,
    }
    appendBwDetailMeta(row, o)
    return row
  }

  const vsNvidia = avg != null ? bwVsNvidiaPercent(avg) : 100
  const row: BwDetailRow = {
    model,
    add: addN,
    copy: copyN,
    scale: scaleN,
    triad: triadN,
    avg,
    vsNvidia,
  }
  appendBwDetailMeta(row, o)
  return row
}

function parseBwCsv(filePath: string): BwDetailRow[] {
  if (!fs.existsSync(filePath)) return []
  const text = fs.readFileSync(filePath, 'utf8').replace(/^\uFEFF/, '')
  const records = parse(text, {
    columns: true,
    skip_empty_lines: true,
    trim: true,
    relax_column_count: true,
  }) as Record<string, unknown>[]
  const out: BwDetailRow[] = []
  for (const rec of records) {
    const o = normalizeCsvRowKeys(rec)
    const row = bwRowFromCsvRecord(o)
    if (row) out.push(row)
  }
  return out
}

/** 访存表：CSV 与首 sheet 的 xlsx，列名经 `readFirstSheetStringRecords` 与 CSV 归一化后一致 */
function parseBwDataFile(filePath: string): BwDetailRow[] {
  if (!filePath.toLowerCase().endsWith('.csv')) {
    const records = readFirstSheetStringRecords(filePath)
    const out: BwDetailRow[] = []
    for (const o of records) {
      const row = bwRowFromCsvRecord(o)
      if (row) out.push(row)
    }
    return out
  }
  return parseBwCsv(filePath)
}

/** 同平台 `{plat}_bw_*.csv` 与 `{plat}_bw_*.xlsx` 取文件名日期较新者；同日优先 CSV */
function pickLatestBwFilePerPlatform(): Record<string, { date: string; full: string }> {
  const csvMap = pickLatestByPlatform(DATA_BW, /^(.+)_bw_(\d+)\.csv$/i)
  const xlsxMap = pickLatestByPlatform(DATA_BW, /^(.+)_bw_(\d+)\.xlsx$/i)
  const plats = new Set([...Object.keys(csvMap), ...Object.keys(xlsxMap)])
  const out: Record<string, { date: string; full: string }> = {}
  for (const plat of plats) {
    const c = csvMap[plat]
    const x = xlsxMap[plat]
    if (!c) out[plat] = x!
    else if (!x) out[plat] = c
    else if (c.date !== x.date) out[plat] = c.date > x.date ? c : x
    else out[plat] = c
  }
  return out
}

function maxBwTableDateIso(rows: BwDetailRow[]): string {
  let max = ''
  for (const r of rows) {
    if (!r.date) continue
    const iso = inferDateToSortable(r.date)
    if (iso && iso > max) max = iso
  }
  return max
}

/** `new_data/bw/bw_template.csv`：多平台同文件，按 platform 列拆成与 `{plat}_bw_*.csv` 相同的结构 */
function parseBwTemplateCsvByPlatform(filePath: string): Record<string, BwDetailRow[]> {
  if (!fs.existsSync(filePath)) return {}
  const text = fs.readFileSync(filePath, 'utf8').replace(/^\uFEFF/, '')
  const records = parse(text, {
    columns: true,
    skip_empty_lines: true,
    trim: true,
    relax_column_count: true,
  }) as Record<string, unknown>[]
  const byPlat: Record<string, BwDetailRow[]> = {}
  for (const rec of records) {
    const o = normalizeCsvRowKeys(rec)
    const platRaw = bwCsvCell(o, 'platform', 'vendor', 'plat', 'chip')
    const plat = normalizePlatform(platRaw === '' ? undefined : platRaw)
    if (!plat) continue
    const row = bwRowFromCsvRecord(o)
    if (!row) continue
    if (!byPlat[plat]) byPlat[plat] = []
    byPlat[plat].push(row)
  }
  return byPlat
}

const CARD_TEMPLATE = {
  openScore: 100,
  ownFw: 'InfiniCore ✦',
  openFw: 'PyTorch',
}

function main() {
  const meta: {
    generatedAt: string
    operatorSources: string[]
    inferSources: string[]
    trainSources: string[]
    commSources: string[]
    bwSources: string[]
    opDatasetUpdatedAt?: string
    inferDatasetUpdatedAt?: string
    trainDatasetUpdatedAt?: string
    commDatasetUpdatedAt?: string
    bwDatasetUpdatedAt?: string
    /** 文件名 `*_train_YYYYMMDD.xlsx` 解析出的日期（行内无 date 时横条回退） */
    trainSourceFileDateByPlatform?: Record<string, string>
    commSourceFileDateByPlatform?: Record<string, string>
    bwSourceFileDateByPlatform?: Record<string, string>
  } = {
    generatedAt: new Date().toISOString(),
    operatorSources: [],
    inferSources: [],
    trainSources: [],
    commSources: [],
    bwSources: [],
  }

  const OP_TABLE_FROM_FILES: Record<string, Record<string, OperatorTableRow[]>> = {}
  const OP_CARD_FROM_FILES: Record<string, Record<string, unknown>> = {}
  const flatAccumulator: Record<string, OperatorFlat[]> = {}

  const datedOp = pickLatestOperatorByPlatform()
  for (const [plat, { full }] of Object.entries(datedOp)) {
    const { byPlat, flatByPlat } = parseOperatorFromFile(full, plat)
    if (parsedHasPlat(byPlat, plat)) {
      OP_TABLE_FROM_FILES[plat] = byPlat[plat]
      meta.operatorSources.push(full.replace(ROOT + path.sep, ''))
      const flats = flatByPlat[plat] || []
      flatAccumulator[plat] = (flatAccumulator[plat] || []).concat(flats)
    }
  }

  const opsDataPath = path.join(DATA_OPERATOR, 'ops_data.csv')
  if (fs.existsSync(opsDataPath)) {
    const { byPlat, flatByPlat } = parseOperatorCsv(opsDataPath, undefined)
    for (const [plat, ops] of Object.entries(byPlat)) {
      if (!OP_TABLE_FROM_FILES[plat]) {
        OP_TABLE_FROM_FILES[plat] = ops
        meta.operatorSources.push(opsDataPath.replace(ROOT + path.sep, ''))
        const flats = flatByPlat[plat] || []
        flatAccumulator[plat] = (flatAccumulator[plat] || []).concat(flats)
      }
    }
  }

  let maxDateSlash = ''
  for (const [plat, flats] of Object.entries(flatAccumulator)) {
    const overview = buildOpPlatformOverview(flats)
    if (overview) {
      const { dataDate, ...cardMetrics } = overview
      OP_CARD_FROM_FILES[plat] = { key: plat, ...CARD_TEMPLATE, ...cardMetrics }
      if (dataDate && dataDate > maxDateSlash) maxDateSlash = dataDate
    }
  }
  if (maxDateSlash) meta.opDatasetUpdatedAt = maxDateSlash.replace(/\//g, '-')

  const INFER_TABLE_FROM_FILES: Record<string, { prefill: InferPrefillRow[]; decode: InferDecodeRow[] }> = {}
  const inferFlatByPlat: Record<string, InferFlatRow[]> = {}
  const datedInfer = pickLatestByPlatform(DATA_INFER, /^(.+)_infer_(\d+)\.csv$/)
  for (const [plat, { full }] of Object.entries(datedInfer)) {
    const { flats, prefill, decode } = parseInferCsvNew(full, plat)
    if (prefill.length || decode.length) {
      INFER_TABLE_FROM_FILES[plat] = { prefill, decode }
      meta.inferSources.push(full.replace(ROOT + path.sep, ''))
      inferFlatByPlat[plat] = flats
    }
  }

  const nvPref = INFER_TABLE_FROM_FILES.nvidia?.prefill || []
  const nvDec = INFER_TABLE_FROM_FILES.nvidia?.decode || []
  for (const plat of Object.keys(INFER_TABLE_FROM_FILES)) {
    const p = INFER_TABLE_FROM_FILES[plat]
    enrichInferWithNvidiaBaseline(p.prefill, p.decode, nvPref, nvDec)
  }

  const INFER_CARD_FROM_FILES: Record<string, Record<string, unknown>> = {}
  let maxInferDate = ''
  for (const [, flats] of Object.entries(inferFlatByPlat)) {
    for (const r of flats) {
      if (r.date) {
        const iso = inferDateToSortable(r.date)
        if (iso && iso > maxInferDate) maxInferDate = iso
      }
    }
  }
  if (maxInferDate) meta.inferDatasetUpdatedAt = maxInferDate

  for (const [plat, pack] of Object.entries(INFER_TABLE_FROM_FILES)) {
    const m = buildInferCardMetrics(pack.prefill, pack.decode)
    if (m) {
      const { dataDate: _drop, ...card } = m
      INFER_CARD_FROM_FILES[plat] = {
        key: plat,
        ownFw: 'Prefill ✦',
        openFw: 'Decode',
        ...card,
      }
    }
  }

  const TRAIN_TABLE_FROM_FILES: Record<string, Omit<TrainTableRow, 'matchKey'>[]> = {}
  const TRAIN_CARD_FROM_FILES: Record<string, Record<string, unknown>> = {}
  const trainByPlat: Record<string, TrainTableRow[]> = {}

  const datedTrain = pickLatestByPlatform(DATA_TRAIN, /^(.+)_train_(\d+)\.xlsx$/i)
  const trainFileDateByPlat: Record<string, string> = {}
  for (const [plat, { date: stamp }] of Object.entries(datedTrain)) {
    const iso = fileStampYmdToIso(stamp)
    if (iso) trainFileDateByPlat[plat] = iso
  }
  if (Object.keys(trainFileDateByPlat).length) meta.trainSourceFileDateByPlatform = trainFileDateByPlat

  for (const [plat, { full }] of Object.entries(datedTrain)) {
    ingestTrainFile(full, plat, trainByPlat)
    if ((trainByPlat[plat] || []).length) {
      meta.trainSources.push(full.replace(ROOT + path.sep, ''))
    }
  }

  enrichTrainBaselines(trainByPlat)

  let maxTrainDate = ''
  for (const rows of Object.values(trainByPlat)) {
    for (const r of rows) {
      if (r.date) {
        const iso = inferDateToSortable(r.date)
        if (iso && iso > maxTrainDate) maxTrainDate = iso
      }
    }
  }
  if (maxTrainDate) meta.trainDatasetUpdatedAt = maxTrainDate

  for (const [plat, rows] of Object.entries(trainByPlat)) {
    if (!rows.length) continue
    TRAIN_TABLE_FROM_FILES[plat] = stripTrainMatchKey(rows)
    const m = buildTrainCardMetrics(rows)
    if (m) {
      TRAIN_CARD_FROM_FILES[plat] = {
        key: plat,
        ownFw: m.ownFwLabel,
        openFw: '',
        openScore: null,
        openVal: null,
        ownScore: m.ownScore,
        ownVal: m.ownVal,
        n: m.n,
        extra: m.extra,
        adv: m.adv,
        advTxt: m.advTxt,
      }
    }
  }

  const COMM_TABLE_FROM_FILES: Record<string, Omit<CommImportRow, 'matchKey'>[]> = {}
  const COMM_CARD_FROM_FILES: Record<string, Record<string, unknown>> = {}
  const commByPlat: Record<string, CommImportRow[]> = {}

  const datedCommXlsx = pickLatestByPlatform(DATA_COMM, /^(.+)_comm_(\d+)\.xlsx$/i)
  const datedCommCsv = pickLatestByPlatform(DATA_COMM, /^(.+)_comm_(\d+)\.csv$/i)
  const commPlats = new Set([...Object.keys(datedCommXlsx), ...Object.keys(datedCommCsv)])
  const commFileDateByPlat: Record<string, string> = {}
  for (const plat of commPlats) {
    const csv = datedCommCsv[plat]
    const xlsx = datedCommXlsx[plat]
    const chosen = csv ?? xlsx
    if (!chosen) continue
    const iso = fileStampYmdToIso(chosen.date)
    if (iso) commFileDateByPlat[plat] = iso
  }
  if (Object.keys(commFileDateByPlat).length) meta.commSourceFileDateByPlatform = commFileDateByPlat

  for (const plat of commPlats) {
    const csv = datedCommCsv[plat]
    const xlsx = datedCommXlsx[plat]
    if (csv) {
      const hint = fileStampYmdToIso(csv.date)
      ingestCommCsvFile(csv.full, plat, commByPlat, hint || undefined)
      if ((commByPlat[plat] || []).length) meta.commSources.push(csv.full.replace(ROOT + path.sep, ''))
    } else if (xlsx) {
      const hint = fileStampYmdToIso(xlsx.date)
      ingestCommFile(xlsx.full, plat, commByPlat, hint || undefined)
      if ((commByPlat[plat] || []).length) meta.commSources.push(xlsx.full.replace(ROOT + path.sep, ''))
    }
  }

  for (const rows of Object.values(commByPlat)) {
    rows.sort((a, b) => {
      if (a.commType !== b.commType) return a.commType.localeCompare(b.commType)
      return a.nGpu - b.nGpu
    })
  }

  enrichCommBaselines(commByPlat)

  let maxCommDate = ''
  for (const rows of Object.values(commByPlat)) {
    for (const r of rows) {
      if (r.date) {
        const iso = inferDateToSortable(r.date)
        if (iso && iso > maxCommDate) maxCommDate = iso
      }
    }
  }
  if (maxCommDate) meta.commDatasetUpdatedAt = maxCommDate

  for (const [plat, rows] of Object.entries(commByPlat)) {
    if (!rows.length) continue
    COMM_TABLE_FROM_FILES[plat] = stripCommMatchKey(rows)
    const m = buildCommCardMetrics(rows)
    if (m) {
      const { overviewOwnFw, overviewOpenFw, ...cardMetrics } = m
      COMM_CARD_FROM_FILES[plat] = {
        key: plat,
        ownFw: overviewOwnFw,
        openFw: overviewOpenFw ?? '',
        ...cardMetrics,
      }
    }
  }

  const BW_TABLE_FROM_FILES: Record<string, BwDetailRow[]> = {}
  const BW_CARD_FROM_FILES: Record<string, Record<string, unknown>> = {}

  const datedBw = pickLatestBwFilePerPlatform()
  const bwFileDateByPlat: Record<string, string> = {}
  for (const [plat, { date: stamp }] of Object.entries(datedBw)) {
    const iso = fileStampYmdToIso(stamp)
    if (iso) bwFileDateByPlat[plat] = iso
  }

  for (const [plat, { full }] of Object.entries(datedBw)) {
    const rows = parseBwDataFile(full)
    if (rows.length) {
      BW_TABLE_FROM_FILES[plat] = rows
      meta.bwSources.push(full.replace(ROOT + path.sep, ''))
      const m = buildBwCardMetrics(rows)
      if (m) {
        BW_CARD_FROM_FILES[plat] = {
          key: plat,
          ownFw: 'HBM均值',
          openFw: '',
          openScore: null,
          openVal: null,
          ...m,
        }
      }
    }
  }

  const bwTemplatePath = path.join(DATA_BW, 'bw_template.csv')
  if (fs.existsSync(bwTemplatePath)) {
    const grouped = parseBwTemplateCsvByPlatform(bwTemplatePath)
    const flatTemplate = Object.values(grouped).flat()
    const templateFileMaxIso = maxBwTableDateIso(flatTemplate)
    for (const [plat, rows] of Object.entries(grouped)) {
      if (!rows.length) continue
      if (!BW_TABLE_FROM_FILES[plat]) {
        BW_TABLE_FROM_FILES[plat] = rows
        meta.bwSources.push(bwTemplatePath.replace(ROOT + path.sep, ''))
        const m = buildBwCardMetrics(rows)
        if (m) {
          BW_CARD_FROM_FILES[plat] = {
            key: plat,
            ownFw: 'HBM均值',
            openFw: '',
            openScore: null,
            openVal: null,
            ...m,
          }
        }
      }
      if (!bwFileDateByPlat[plat]) {
        const rowIso = maxBwTableDateIso(rows)
        bwFileDateByPlat[plat] = rowIso || templateFileMaxIso
      }
    }
  }

  if (Object.keys(bwFileDateByPlat).length) meta.bwSourceFileDateByPlatform = bwFileDateByPlat

  let maxBwDate = ''
  for (const rows of Object.values(BW_TABLE_FROM_FILES)) {
    for (const r of rows) {
      if (r.date) {
        const iso = inferDateToSortable(r.date)
        if (iso && iso > maxBwDate) maxBwDate = iso
      }
    }
  }
  if (maxBwDate) meta.bwDatasetUpdatedAt = maxBwDate

  const ts = `/* eslint-disable */
// 本文件由 npm run generate:data 自动生成，请勿手改。算子：new_data/operator；推理：new_data/infer；训练：new_data/train；通信：new_data/comm；访存：new_data/bw。

export const OP_TABLE_FROM_FILES = ${JSON.stringify(OP_TABLE_FROM_FILES, null, 2)} as Record<string, Record<string, { shape: string; dtype: string; ic: number; pt: number; remarks: string; scoreEligible: boolean; date?: string }[]>>

export const OP_CARD_FROM_FILES = ${JSON.stringify(OP_CARD_FROM_FILES, null, 2)} as Record<string, Record<string, unknown>>

export const INFER_TABLE_FROM_FILES = ${JSON.stringify(INFER_TABLE_FROM_FILES, null, 2)} as Record<string, { prefill: Array<Record<string, unknown>>; decode: Array<Record<string, unknown>> }>

export const INFER_CARD_FROM_FILES = ${JSON.stringify(INFER_CARD_FROM_FILES, null, 2)} as Record<string, Record<string, unknown>>

export const TRAIN_TABLE_FROM_FILES = ${JSON.stringify(TRAIN_TABLE_FROM_FILES, null, 2)} as Record<string, Array<Record<string, unknown>>>

export const TRAIN_CARD_FROM_FILES = ${JSON.stringify(TRAIN_CARD_FROM_FILES, null, 2)} as Record<string, Record<string, unknown>>

export const COMM_TABLE_FROM_FILES = ${JSON.stringify(COMM_TABLE_FROM_FILES, null, 2)} as Record<string, Array<Record<string, unknown>>>

export const COMM_CARD_FROM_FILES = ${JSON.stringify(COMM_CARD_FROM_FILES, null, 2)} as Record<string, Record<string, unknown>>

export const BW_TABLE_FROM_FILES = ${JSON.stringify(BW_TABLE_FROM_FILES, null, 2)} as Record<string, Array<Record<string, unknown>>>

export const BW_CARD_FROM_FILES = ${JSON.stringify(BW_CARD_FROM_FILES, null, 2)} as Record<string, Record<string, unknown>>

export const BENCHMARK_DATA_META = ${JSON.stringify(meta, null, 2)}
`

  fs.mkdirSync(path.dirname(OUT_FILE), { recursive: true })
  fs.writeFileSync(OUT_FILE, ts, 'utf8')
  console.log('Wrote', OUT_FILE)
  console.log('Platforms (op):', Object.keys(OP_TABLE_FROM_FILES).join(', ') || '(none)')
  console.log('Platforms (infer):', Object.keys(INFER_TABLE_FROM_FILES).join(', ') || '(none)')
  console.log('Platforms (train):', Object.keys(TRAIN_TABLE_FROM_FILES).join(', ') || '(none)')
  console.log('Platforms (comm):', Object.keys(COMM_TABLE_FROM_FILES).join(', ') || '(none)')
  console.log('Platforms (bw):', Object.keys(BW_TABLE_FROM_FILES).join(', ') || '(none)')
}

function parsedHasPlat(byPlat: Record<string, Record<string, OperatorTableRow[]>>, plat: string) {
  return byPlat[plat] && Object.keys(byPlat[plat]).length > 0
}

main()
