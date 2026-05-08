/**
 * 从仓库根目录 `data/operator/*.csv`、`data/infer/*.csv` 生成 `src/data/generatedFromFiles.ts`，
 * 供 dashboardConfig 合并后替换对应平台上的详情表数据。
 */
import { parse } from 'csv-parse/sync'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const ROOT = path.join(__dirname, '..', '..')
const DATA_OPERATOR = path.join(ROOT, 'data', 'operator')
const DATA_INFER = path.join(ROOT, 'data', 'infer')
const OUT_FILE = path.join(__dirname, '..', 'src', 'data', 'generatedFromFiles.ts')

/** CSV 中的平台前缀 → 仪表盘 PLATFORMS.key */
const PLATFORM_ALIAS = {
  moore: 'mthreads',
  ali: 'generic',
}

function normalizePlatform(raw) {
  const k = String(raw || '').trim().toLowerCase()
  return PLATFORM_ALIAS[k] ?? k
}

function parseNum(v) {
  if (v == null || v === '') return NaN
  const s = String(v).trim().toUpperCase()
  if (s === 'NA' || s === 'N/A') return NaN
  const n = Number.parseFloat(String(v))
  return Number.isFinite(n) ? n : NaN
}

/** op_name → 与 dashboard DIMS pills / OP_TABLE 键一致 */
function opNameToKey(name) {
  const lower = String(name || '').trim().toLowerCase().replace(/\s+/g, '')
  const map = {
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

function normalizeShape(s) {
  return String(s || '')
    .trim()
    .replace(/^"|"$/g, '')
    .replace(/,\s*/g, ', ')
}

/**
 * 解析单个 operator CSV。
 * `forcedPlatform`：来自文件名 `{platform}_operator_*.csv`，优先于表内 platform 列（避免填错列）。
 */
function parseOperatorCsv(filePath, forcedPlatform) {
  const text = fs.readFileSync(filePath, 'utf8').replace(/^\uFEFF/, '')
  const records = parse(text, {
    columns: true,
    skip_empty_lines: true,
    trim: true,
    relax_column_count: true,
  })
  const byPlat = {}
  for (const row of records) {
    const plat = forcedPlatform ?? normalizePlatform(row.platform)
    if (!plat) continue
    const opKey = opNameToKey(row.op_name)
    const ic = parseNum(row.ic_latency_ms)
    const pt = parseNum(row.pt_latency_ms)
    if (!Number.isFinite(ic) || !Number.isFinite(pt)) continue
    const dtype = String(row.dtype || '').toUpperCase()
    const shape = normalizeShape(row.shape_config)
    if (!byPlat[plat]) byPlat[plat] = {}
    if (!byPlat[plat][opKey]) byPlat[plat][opKey] = []
    byPlat[plat][opKey].push({ shape, dtype, ic, pt })
  }
  return byPlat
}

function parseInferCsv(filePath, forcedPlatform) {
  const text = fs.readFileSync(filePath, 'utf8').replace(/^\uFEFF/, '')
  const rows = parse(text, {
    relax_column_count: true,
    skip_empty_lines: true,
    bom: true,
  })
  if (rows.length < 2) return {}

  const prefillByPlat = {}
  const decodeByPlat = {}

  for (let i = 1; i < rows.length; i++) {
    const cols = rows[i].map((c) => String(c ?? '').trim())
    if (cols.length < 12) continue
    const plat = forcedPlatform ?? normalizePlatform(cols[0])
    if (!plat) continue
    const model = cols[1]
    const batch = Number(cols[2])
    const inLen = Number(cols[3])
    const outTok = Number(cols[4])
    const outLen = Number.isFinite(outTok) ? outTok : 128
    const ttft = parseNum(cols[7])
    const prefillTps = parseNum(cols[10])
    const decodeTps = parseNum(cols[11])

    if (!prefillByPlat[plat]) prefillByPlat[plat] = []
    if (!decodeByPlat[plat]) decodeByPlat[plat] = []

    if (Number.isFinite(prefillTps)) {
      const row = {
        batch,
        inLen,
        outLen,
        model,
        tps: Math.round(prefillTps),
        framework: 'InfiniLM',
      }
      if (Number.isFinite(ttft)) row.ttft = ttft
      prefillByPlat[plat].push(row)
    }
    if (Number.isFinite(decodeTps)) {
      decodeByPlat[plat].push({
        batch,
        inLen,
        outLen,
        model,
        tps: Math.round(decodeTps),
        framework: 'InfiniLM',
      })
    }
  }

  const out = {}
  const plats = new Set([...Object.keys(prefillByPlat), ...Object.keys(decodeByPlat)])
  for (const p of plats) {
    out[p] = {
      prefill: prefillByPlat[p] || [],
      decode: decodeByPlat[p] || [],
    }
  }
  return out
}

function pickLatestByPlatform(dir, pattern) {
  const map = {}
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

function main() {
  const meta = {
    generatedAt: new Date().toISOString(),
    operatorSources: [],
    inferSources: [],
  }

  const OP_TABLE_FROM_FILES = {}

  const datedOp = pickLatestByPlatform(DATA_OPERATOR, /^(.+)_operator_(\d+)\.csv$/)
  for (const [plat, { full }] of Object.entries(datedOp)) {
    const parsed = parseOperatorCsv(full, plat)
    if (parsed[plat]) {
      OP_TABLE_FROM_FILES[plat] = parsed[plat]
      meta.operatorSources.push(full.replace(ROOT + path.sep, ''))
    }
  }

  const opsDataPath = path.join(DATA_OPERATOR, 'ops_data.csv')
  if (fs.existsSync(opsDataPath)) {
    const parsed = parseOperatorCsv(opsDataPath)
    for (const [plat, ops] of Object.entries(parsed)) {
      if (!OP_TABLE_FROM_FILES[plat]) {
        OP_TABLE_FROM_FILES[plat] = ops
        meta.operatorSources.push(opsDataPath.replace(ROOT + path.sep, ''))
      }
    }
  }

  const INFER_TABLE_FROM_FILES = {}
  const datedInfer = pickLatestByPlatform(DATA_INFER, /^(.+)_infer_(\d+)\.csv$/)
  for (const [plat, { full }] of Object.entries(datedInfer)) {
    const parsed = parseInferCsv(full, plat)
    if (parsed[plat]) {
      INFER_TABLE_FROM_FILES[plat] = parsed[plat]
      meta.inferSources.push(full.replace(ROOT + path.sep, ''))
    }
  }

  const ts = `/* eslint-disable */
// 本文件由 npm run generate:data 自动生成，请勿手改。源文件：仓库根目录 data/operator、data/infer 下 CSV。

export const OP_TABLE_FROM_FILES = ${JSON.stringify(OP_TABLE_FROM_FILES, null, 2)} as Record<string, Record<string, { shape: string; dtype: string; ic: number; pt: number }[]>>

export const INFER_TABLE_FROM_FILES = ${JSON.stringify(INFER_TABLE_FROM_FILES, null, 2)} as Record<string, { prefill: Array<Record<string, unknown>>; decode: Array<Record<string, unknown>> }>

export const BENCHMARK_DATA_META = ${JSON.stringify(meta, null, 2)}
`

  fs.mkdirSync(path.dirname(OUT_FILE), { recursive: true })
  fs.writeFileSync(OUT_FILE, ts, 'utf8')
  console.log('Wrote', OUT_FILE)
  console.log('Platforms (op):', Object.keys(OP_TABLE_FROM_FILES).join(', ') || '(none)')
  console.log('Platforms (infer):', Object.keys(INFER_TABLE_FROM_FILES).join(', ') || '(none)')
}

main()
