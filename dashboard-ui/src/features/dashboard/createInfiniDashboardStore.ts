import { Modal } from 'ant-design-vue'
import { computed, ref } from 'vue'
import {
  BENCHMARK_DATA_META,
  BW_TABLE,
  CARD_DATA,
  COMM_TABLE,
  DIMS,
  getCiSeriesForDim,
  INFER_TABLE,
  OP_TABLE,
  PLATFORMS,
  TRAIN_TABLE,
  type BwDetailRow,
  type CommDetailRow,
  type TrainDetailRow,
} from '@/data'
import {
  applyCardFilter,
  avgOpScore,
  parseLatencyMs,
  type CardRow,
} from '@/features/dashboard/dashboardFilterHelpers'
import { canComputeOpRowScore } from '@/features/dashboard/operatorBenchmark'
import {
  alignInferSeries,
  filterInferRows,
} from '@/features/dashboard/inferBenchmark'
import { filterCommRows } from '@/features/dashboard/commBenchmark'
import { BW_NVIDIA_BASELINE_GBPS, pickBestBwRow } from '@/features/dashboard/bwBenchmark'
import { filterTrainRows } from '@/features/dashboard/trainBenchmark'
import {
  buildBwTestEnvLine,
  buildCommTestEnvLine,
  buildInferTestEnvLine,
  buildOpTestEnvLine,
  buildTrainTestEnvLine,
  DETAIL_TEST_ENV_SOURCE_HINT,
} from '@/features/dashboard/detailTestEnvBar'
import {
  buildBwBarAvg,
  buildBwBarModes,
  buildCiLineOption,
  buildCommBarBw,
  buildCommBarVs,
  buildCompareLatencyBar,
  buildCompareScoreBar,
  buildInferDecodeBarAligned,
  buildInferPrefillBarAligned,
  buildOpBarAvgOption,
  buildOpLineOption,
  buildTrainBarThroughput,
  buildTrainBarVs,
} from '@/utils/echartsInfini'

export type MainView = 'overview' | 'detail' | 'compare'

const OTABLE = OP_TABLE as Record<
  string,
  Record<
    string,
    {
      shape: string
      dtype: string
      ic: number
      pt: number
      remarks?: string
      scoreEligible?: boolean
      date?: string
    }[]
  >
>

type InferRowCore = {
  configKey: string
  batch: number
  inLen: number
  outLen: number
  model: string
  dtype?: string
  tps: number
  ttft?: number
  decodeLatencyMs?: number
  vsNvidia?: number | null
  nvidiaBaselineTps?: number | null
  nGpu?: number
  remarks?: string
  date?: string
}

const ITABLE = INFER_TABLE as unknown as Record<
  string,
  { prefill: InferRowCore[]; decode: InferRowCore[] } | undefined
>

type TrainRow = TrainDetailRow

const TTRAIN = TRAIN_TABLE as Record<string, TrainRow[] | undefined>

const CTABLE = COMM_TABLE as Record<string, CommDetailRow[] | undefined>

const BTABLE = BW_TABLE as Record<string, BwDetailRow[] | undefined>

type BenchMetaFileDates = {
  trainSourceFileDateByPlatform?: Record<string, string>
  commSourceFileDateByPlatform?: Record<string, string>
  bwSourceFileDateByPlatform?: Record<string, string>
}

function benchSourceFileDate(plat: string, dim: keyof BenchMetaFileDates): string | undefined {
  const m = BENCHMARK_DATA_META as BenchMetaFileDates
  return m[dim]?.[plat]
}

export function createInfiniDashboardStore() {
  const selectedPlatKeys = ref<string[]>(PLATFORMS.map((p) => p.key))
  const comparePlatKeys = ref<string[]>(['nvidia'])
  const currentView = ref<MainView>('overview')
  const activeDim = ref(0)
  const filterState = ref<Record<string, Record<number, number> | undefined>>({})
  const sortDesc = ref(true)
  const detailState = ref({
    platKey: 'nvidia',
    opKey: 'CausalSoftmax',
    prec: '全部',
    inferTab: 'prefill' as 'prefill' | 'decode',
  })
  const bcBrand = ref('')
  const bcDim = ref('')
  const detailTableTab = ref<'data' | 'score'>('data')
  const ciTabKey = ref('ci-stats')

  const activeDimKey = computed(() => DIMS[activeDim.value].key)

  const overviewCards = computed(() => {
    const dim = DIMS[activeDim.value]
    const raw = (CARD_DATA as Record<string, CardRow[]>)[dim.key] || []
    let data = raw.filter((c) => selectedPlatKeys.value.includes(c.key))
    data = applyCardFilter(data, activeDim.value, filterState.value) as CardRow[]
    const sorted = [...data].sort((a, b) => {
      const va = a.ownScore ?? 0
      const vb = b.ownScore ?? 0
      const primary = sortDesc.value ? vb - va : va - vb
      if (primary !== 0) return primary
      return String(a.key).localeCompare(String(b.key))
    })
    return sorted
  })

  const compareCards = computed(() => {
    const dim = DIMS[activeDim.value]
    const raw = (CARD_DATA as Record<string, CardRow[]>)[dim.key] || []
    return raw.filter((c) => comparePlatKeys.value.includes(c.key))
  })

  const detailPlat = computed(() => PLATFORMS.find((p) => p.key === detailState.value.platKey)!)

  const detailCard = computed(() => {
    const dim = DIMS[activeDim.value]
    const rows = (CARD_DATA as Record<string, CardRow[]>)[dim.key] || []
    return rows.find((c) => c.key === detailState.value.platKey) || ({} as CardRow)
  })

  /** 算子详情表行 */
  const opDetailRows = computed(() => {
    const platOps = OTABLE[detailState.value.platKey] || {}
    const rows = platOps[detailState.value.opKey] || []
    if (detailState.value.prec === '全部') return rows
    return rows.filter((r) => r.dtype === detailState.value.prec)
  })

  function inferPills() {
    const dim = DIMS.find((d) => d.key === 'infer')!
    const fs = filterState.value.infer || {}
    const batchPill = dim.filters[0]?.pills[fs[0] ?? 0]
    const inPill = dim.filters[1]?.pills[fs[1] ?? 0]
    return { batchPill, inPill }
  }

  const inferPrefillFiltered = computed(() => {
    const plat = detailState.value.platKey
    const { batchPill, inPill } = inferPills()
    return filterInferRows(ITABLE[plat]?.prefill || [], batchPill, inPill)
  })

  const inferDecodeFiltered = computed(() => {
    const plat = detailState.value.platKey
    const { batchPill, inPill } = inferPills()
    return filterInferRows(ITABLE[plat]?.decode || [], batchPill, inPill)
  })

  const inferNvidiaPrefillFiltered = computed(() => {
    const { batchPill, inPill } = inferPills()
    return filterInferRows(ITABLE.nvidia?.prefill || [], batchPill, inPill)
  })

  const inferNvidiaDecodeFiltered = computed(() => {
    const { batchPill, inPill } = inferPills()
    return filterInferRows(ITABLE.nvidia?.decode || [], batchPill, inPill)
  })

  const inferDetailTabRows = computed(() =>
    detailState.value.inferTab === 'prefill'
      ? inferPrefillFiltered.value
      : inferDecodeFiltered.value,
  )

  const inferNvidiaTabRows = computed(() =>
    detailState.value.inferTab === 'prefill'
      ? inferNvidiaPrefillFiltered.value
      : inferNvidiaDecodeFiltered.value,
  )

  function trainFrameworkPill() {
    const dim = DIMS.find((d) => d.key === 'train')!
    const fs = filterState.value.train || {}
    return dim.filters[0]?.pills[fs[0] ?? 0]
  }

  const trainDetailRows = computed(() => {
    const plat = detailState.value.platKey
    const pill = trainFrameworkPill()
    return filterTrainRows(TTRAIN[plat], pill)
  })

  const commNvidiaBaselineRows = computed(() => CTABLE.nvidia || [])

  const commDetailRows = computed(() => {
    const plat = detailState.value.platKey
    const dim = DIMS.find((d) => d.key === 'comm')!
    const fs = filterState.value.comm || {}
    const pill = dim.filters[0]?.pills[fs[0] ?? 0]
    return filterCommRows(CTABLE[plat], pill)
  })

  const bwNvidiaRefRow = computed(() => {
    const rows = BTABLE.nvidia || []
    return pickBestBwRow(rows) ?? rows[0] ?? null
  })

  const detailTestEnvLine = computed(() => {
    const dk = activeDimKey.value
    const plat = detailState.value.platKey
    if (dk === 'infer') {
      const pack = ITABLE[plat]
      const inferWide = [...(pack?.prefill ?? []), ...(pack?.decode ?? [])]
      return buildInferTestEnvLine(
        inferDetailTabRows.value as InferRowCore[],
        inferWide,
      )
    }
    if (dk === 'op') {
      const platOps = OTABLE[plat] || {}
      const platformWide = Object.values(platOps).flat() as { date?: string }[]
      return buildOpTestEnvLine(plat, opDetailRows.value, platformWide)
    }
    if (dk === 'train') {
      return buildTrainTestEnvLine(
        plat,
        trainDetailRows.value,
        TTRAIN[plat] ?? [],
        benchSourceFileDate(plat, 'trainSourceFileDateByPlatform'),
      )
    }
    if (dk === 'comm') {
      return buildCommTestEnvLine(
        plat,
        commDetailRows.value,
        CTABLE[plat] ?? [],
        benchSourceFileDate(plat, 'commSourceFileDateByPlatform'),
      )
    }
    if (dk === 'bw') {
      return buildBwTestEnvLine(
        plat,
        BTABLE[plat] || [],
        benchSourceFileDate(plat, 'bwSourceFileDateByPlatform'),
      )
    }
    return '—'
  })

  const detailTestEnvSourceHint = computed(() => DETAIL_TEST_ENV_SOURCE_HINT)

  const lineChartOption = computed(() => {
    const plat = detailPlat.value
    const dk = activeDimKey.value
    if (dk === 'op') {
      const rows = opDetailRows.value
      if (!rows.length) return {}
      return buildOpLineOption(rows, plat.color)
    }
    if (dk === 'infer') {
      const pre = inferPrefillFiltered.value
      const nv = inferNvidiaPrefillFiltered.value
      if (!pre.length && !nv.length) return {}
      const al = alignInferSeries(pre, nv)
      return buildInferPrefillBarAligned(al.categories, al.platVals, al.nvVals, plat.color)
    }
    if (dk === 'train') {
      const rows = trainDetailRows.value
      if (!rows?.length) return {}
      return buildTrainBarThroughput(rows, plat.color)
    }
    if (dk === 'comm') {
      const rows = commDetailRows.value
      if (!rows?.length) return {}
      return buildCommBarBw(rows, plat.color)
    }
    if (dk === 'bw') {
      const rows = BTABLE[detailState.value.platKey] || []
      const filtered = rows.filter((r) => r.avg != null)
      if (!filtered.length) return {}
      return buildBwBarAvg(filtered, plat.color, BW_NVIDIA_BASELINE_GBPS)
    }
    return {}
  })

  const barChartOption = computed(() => {
    const plat = detailPlat.value
    const dk = activeDimKey.value
    if (dk === 'op') {
      const platOps = OTABLE[detailState.value.platKey] || {}
      const opKeys = Object.keys(platOps)
      const scores = opKeys.map((op) => {
        const rs = platOps[op] || []
        const valid = rs.filter((r) =>
          canComputeOpRowScore(r.ic, r.pt, r.remarks ?? ''),
        )
        return valid.length
          ? Math.round(valid.reduce((a, r) => a + (r.pt / r.ic) * 100, 0) / valid.length)
          : 0
      })
      return buildOpBarAvgOption(opKeys, scores, plat.color)
    }
    if (dk === 'infer') {
      const de = inferDecodeFiltered.value
      const nvDe = inferNvidiaDecodeFiltered.value
      if (!de.length && !nvDe.length) return {}
      const al = alignInferSeries(de, nvDe)
      return buildInferDecodeBarAligned(al.categories, al.platVals, al.nvVals, plat.color)
    }
    if (dk === 'train') {
      const rows = trainDetailRows.value
      if (!rows?.length) return {}
      return buildTrainBarVs(rows)
    }
    if (dk === 'comm') {
      const rows = commDetailRows.value
      if (!rows?.length) return {}
      return buildCommBarVs(rows)
    }
    if (dk === 'bw') {
      const rows = BTABLE[detailState.value.platKey] || []
      const best = pickBestBwRow(rows)
      const nvidiaRow = bwNvidiaRefRow.value
      if (!best || !nvidiaRow) return {}
      return buildBwBarModes(best, nvidiaRow, plat.color)
    }
    return {}
  })

  const ciChartOption = computed(() => buildCiLineOption(getCiSeriesForDim(activeDimKey.value)))

  const compareScoreOption = computed(() => {
    const cards = compareCards.value
    const plats = cards.map((c) => PLATFORMS.find((p) => p.key === c.key)!).filter(Boolean)
    if (!cards.length || !plats.length) return {}
    return buildCompareScoreBar(cards as { key: string; ownScore: number | null }[], plats)
  })

  const compareLatencyOption = computed(() => {
    const cards = compareCards.value.filter((c) => parseLatencyMs(c.ownVal as string) != null)
    const names = cards.map((c) => PLATFORMS.find((p) => p.key === c.key)!.name)
    const latencies = cards.map((c) => parseLatencyMs(c.ownVal as string) as number)
    const colors = cards.map((c) => PLATFORMS.find((p) => p.key === c.key)!.color)
    if (!latencies.length) return {}
    return buildCompareLatencyBar(names, latencies, colors)
  })

  const detailTitle = computed(() => {
    const plat = detailPlat.value
    const dim = DIMS[activeDim.value]
    return `${plat.name} · ${dim.label}详情`
  })

  const tableNotice = computed(() => {
    const k = activeDimKey.value
    if (k === 'op')
      return '每行得分 = PyTorch延迟 ÷ InfiniCore延迟 × 100（同 shape + dtype）· >100 自研更快；备注含 failed / Device Type Not Supported 时 InfiniCore 延迟不参与计分，得分显示为「—」· ✦ InfiniCore 为自研框架'
    if (k === 'infer')
      return 'Prefill / Decode 吞吐（tokens/s）· 相对 NVIDIA 同 batch+in-len+out-len 配置；顶栏 Batch / In-len 与表格、KPI、柱状图联动取交集'
    if (k === 'train')
      return '训练吞吐 tpps = tokens per process per second · 相对 NVIDIA 同 (framework, model, n_gpu, seq_len, dtype) 配置；顶栏「框架」与表格、KPI、柱状图联动'
    if (k === 'comm')
      return '带宽 GB/s · 行级 vs NVIDIA = 同 (comm_type, n_gpu) 下本机 bw ÷ NVIDIA bw ×100；顶栏「通信类型」与表格、KPI、柱状图联动'
    if (k === 'bw')
      return '访存带宽 GB/s · 行级 vs NVIDIA = bw_GBps÷1607.4561×100；详情表列出全部型号；折线/柱图均值对比第二根柱为 NVIDIA A100 固定基线；四模式柱取本平台 MAX(bw_GBps) 行 vs NVIDIA 参考行'
    return ''
  })

  const scoreTabHint = computed(() => {
    const k = activeDimKey.value
    const msgs: Record<string, string> = {
      op: '每行得分 = PyTorch延迟 ÷ InfiniCore延迟 × 100；备注触发失败规则时该行不参与计分（与 CSV 规格一致）。柱状图为各算子类型下合格行的平均得分。',
      infer:
        'Prefill / Decode 吞吐（tokens/s）· vs NVIDIA 同配置百分比见表；顶栏筛选与表格、KPI、图表联动（交集）',
      train:
        '训练吞吐 tpps · vs NVIDIA 同配置百分比见表；顶栏「框架」筛选与表格、KPI、图表联动',
      comm: '带宽 GB/s · vs NVIDIA 按 (comm_type, n_gpu) 对齐；顶栏筛选与表格、KPI、图表联动',
      bw: '详情表列出该平台全部型号；vs NVIDIA = bw_GBps÷1607.4561×100（≥100 与 <100 分色）；KPI 与柱状图取 MAX(bw_GBps) 所在行，四模式对比柱与 NVIDIA A100 四模式参考行对齐',
    }
    return msgs[k] || ''
  })

  const comparePageTitle = computed(() => `${DIMS[activeDim.value].label} · 平台横向对比`)

  const comparePageSubtitle = computed(() =>
    compareCards.value
      .map((c) => PLATFORMS.find((p) => p.key === c.key)?.name || '')
      .filter(Boolean)
      .join(' vs '),
  )

  const compareKpiBlocks = computed(() => {
    const cards = compareCards.value
    const bestScore = Math.max(...cards.map((c) => (c.ownScore ?? 0) || 0))
    const sorted = [...cards].sort((a, b) => (b.ownScore ?? 0) - (a.ownScore ?? 0))
    return cards.map((c) => {
      const p = PLATFORMS.find((x) => x.key === c.key)!
      const rank = sorted.findIndex((x) => x.key === c.key) + 1
      const isBest = c.ownScore === bestScore
      return { card: c, plat: p, rank, isBest }
    })
  })

  const compareTableRows = computed(() => {
    const cards = compareCards.value
    const base = cards.find((x) => x.key === 'nvidia')
    return cards.map((c) => {
      const p = PLATFORMS.find((x) => x.key === c.key)!
      const delta =
        base?.ownScore != null && base.ownScore !== 0
          ? Math.round(((c.ownScore ?? 0) - base.ownScore) / base.ownScore * 100)
          : null
      return { plat: p, card: c, delta }
    })
  })

  /** 详情顶栏 4 格 KPI（与 HTML renderKPI 一致） */
  const detailKpiCells = computed(() => {
    const platKey = detailState.value.platKey
    const dimKey = activeDimKey.value
    const c = detailCard.value as CardRow & {
      ownVal?: string
      n?: number
      extra?: string
      ownScore?: number | null
    }
    if (dimKey === 'op') {
      return [
        { val: c.ownScore ?? '—', lbl: '最优得分', sub: 'InfiniCore vs PyTorch（全量测试中最高分所在行）' },
        { val: c.n ?? '—', lbl: '测试记录', sub: c.extra || '' },
        { val: c.ownVal ?? '—', lbl: 'InfiniCore 代表延迟', sub: '自研框架', valSm: true },
        { val: c.openVal ?? '—', lbl: 'PyTorch 代表延迟', sub: '开源基准', valSm: true, muted: true },
      ]
    }
    if (dimKey === 'infer') {
      const prefill = inferPrefillFiltered.value
      const decode = inferDecodeFiltered.value
      const peakPrefill = prefill.length ? Math.max(...prefill.map((r) => r.tps)) : null
      const peakDecode = decode.length ? Math.max(...decode.map((r) => r.tps)) : null
      const minTTFT = prefill.filter((r) => r.ttft).length
        ? Math.min(...prefill.filter((r) => r.ttft).map((r) => r.ttft!))
        : null
      const argPref = prefill.length
        ? prefill.reduce((a, b) => (a.tps >= b.tps ? a : b))
        : null
      const model = argPref?.model || prefill[0]?.model || c.extra || ''
      const cfgSub = argPref ? `batch=${argPref.batch} in=${argPref.inLen}` : ''
      const nConfigs = new Set([...prefill.map((r) => r.configKey), ...decode.map((r) => r.configKey)])
      return [
        {
          val: peakPrefill != null ? peakPrefill.toLocaleString() : c.ownVal || '—',
          lbl: 'Prefill 峰值 TPS',
          sub: `InfiniLM · ${model}${cfgSub ? ' · ' + cfgSub : ''}`,
          valSm: true,
        },
        {
          val: peakDecode != null ? peakDecode.toLocaleString() : '—',
          lbl: 'Decode 峰值 TPS',
          sub: `InfiniLM · ${model}`,
          valSm: true,
        },
        {
          val: minTTFT != null ? minTTFT + 'ms' : '—',
          lbl: '最低 TTFT',
          sub: '首 token 延迟（当前筛选下）',
          valSm: true,
        },
        {
          val: (nConfigs.size || c.n || '—') as string | number,
          lbl: '测试配置数',
          sub: 'Prefill ∪ Decode 配置键',
        },
      ]
    }
    if (dimKey === 'train') {
      const rows = trainDetailRows.value
      const best = rows?.length ? rows.reduce((a, b) => (a.tps > b.tps ? a : b)) : undefined
      return [
        {
          val: best?.tps ? best.tps.toLocaleString() + ' tpps' : c.ownVal || '—',
          lbl: '最高训练吞吐',
          sub: `${best?.framework || ''} · ${best?.model || ''}`,
          valSm: true,
        },
        {
          val: best?.vsA100 != null ? best.vsA100 + '%' : '—',
          lbl: 'vs NVIDIA 基线',
          sub: '同配置吞吐比',
          valSm: true,
        },
        { val: rows?.length || '—', lbl: '测试条数', sub: c.extra || '' },
        {
          val: best?.parallel || '—',
          lbl: '代表配置',
          sub: best?.dtype || '',
          valSm: true,
        },
      ]
    }
    if (dimKey === 'comm') {
      const rows = commDetailRows.value
      const p2p = rows?.find((r) => r.commType === 'p2p')
      const ar = rows?.find((r) => r.commType === 'allreduce')
      return [
        {
          val: p2p ? p2p.bw + ' GB/s' : '—',
          lbl: 'P2P 带宽',
          sub: p2p ? p2p.linkType + ' · ' + p2p.nGpu + ' GPU' : '',
          valSm: true,
        },
        {
          val: ar ? ar.bw + ' GB/s' : '—',
          lbl: 'AllReduce 带宽',
          sub: ar ? ar.linkType + ' · ' + ar.nGpu + ' GPU' : '',
          valSm: true,
        },
        {
          val: p2p ? p2p.vsA100 + '%' : '—',
          lbl: 'P2P vs NVIDIA',
          sub: '同 comm_type + n_gpu 基线',
          valSm: true,
        },
        {
          val: ar ? ar.vsA100 + '%' : '—',
          lbl: 'AllReduce vs NVIDIA',
          sub: '同 comm_type + n_gpu 基线',
          valSm: true,
        },
      ]
    }
    if (dimKey === 'bw') {
      const rows = BTABLE[platKey] || []
      const best = pickBestBwRow(rows)
      return [
        {
          val: best?.avg != null ? best.avg.toFixed(1) + ' GB/s' : '—',
          lbl: 'HBM 均值带宽',
          sub: `${best?.model || ''} · 4模式均值`,
          valSm: true,
        },
        {
          val: best != null ? best.vsNvidia + '%' : '—',
          lbl: 'vs NVIDIA',
          sub: `A100 基线 ${BW_NVIDIA_BASELINE_GBPS} GB/s`,
          valSm: true,
        },
        {
          val: best?.add != null ? best.add.toFixed(1) : '—',
          lbl: 'add GB/s',
          sub: '读写混合模式',
          valSm: true,
        },
        {
          val: best?.triad != null ? best.triad.toFixed(1) : '—',
          lbl: 'triad GB/s',
          sub: '综合压力模式',
          valSm: true,
        },
      ]
    }
    return []
  })

  function togglePlat(key: string) {
    const i = selectedPlatKeys.value.indexOf(key)
    if (i >= 0) selectedPlatKeys.value = selectedPlatKeys.value.filter((k) => k !== key)
    else selectedPlatKeys.value = [...selectedPlatKeys.value, key]
  }

  function selectAll() {
    const allSel = PLATFORMS.every((p) => selectedPlatKeys.value.includes(p.key))
    selectedPlatKeys.value = allSel ? [] : PLATFORMS.map((p) => p.key)
  }

  function selectDomestic() {
    selectedPlatKeys.value = PLATFORMS.filter((p) => p.domestic).map((p) => p.key)
  }

  function setDim(i: number) {
    activeDim.value = i
    filterState.value = {}
    switchMainView('overview')
  }

  function setFilter(fi: number, pi: number) {
    const k = DIMS[activeDim.value].key
    if (!filterState.value[k]) filterState.value[k] = {}
    filterState.value[k]![fi] = pi
    const dim = DIMS[activeDim.value]
    const pill = dim.filters[fi]?.pills[pi]
    if (currentView.value === 'detail' && k === 'op') {
      if (fi === 0 && pill && pill !== '全部') detailState.value.opKey = pill
      if (fi === 1 && pill) detailState.value.prec = pill
    }
  }

  function toggleSort() {
    sortDesc.value = !sortDesc.value
  }

  function toggleCompare(key: string) {
    const idx = comparePlatKeys.value.indexOf(key)
    if (idx >= 0) comparePlatKeys.value = comparePlatKeys.value.filter((k) => k !== key)
    else {
      if (comparePlatKeys.value.length >= 4) {
        Modal.warning({ title: '提示', content: '最多选择 4 个平台进行对比' })
        return
      }
      comparePlatKeys.value = [...comparePlatKeys.value, key]
    }
  }

  function clearCompare() {
    comparePlatKeys.value = []
  }

  function switchMainView(v: MainView) {
    currentView.value = v
  }

  function opTagKeys(): string[] {
    const platOps = OTABLE[detailState.value.platKey] || {}
    const ops = Object.keys(platOps).length ? Object.keys(platOps) : ['全部数据']
    if (!platOps[detailState.value.opKey]) detailState.value.opKey = ops[0]
    return ops
  }

  function setOp(op: string) {
    detailState.value.opKey = op
  }

  function setPrec(p: string) {
    detailState.value.prec = p
  }

  function setInferTab(t: 'prefill' | 'decode') {
    detailState.value.inferTab = t
  }

  return {
    PLATFORMS,
    DIMS,
    CARD_DATA,
    OP_TABLE: OTABLE,
    INFER_TABLE,
    TRAIN_TABLE,
    COMM_TABLE,
    BW_TABLE,
    selectedPlatKeys,
    comparePlatKeys,
    currentView,
    activeDim,
    filterState,
    sortDesc,
    detailState,
    bcBrand,
    bcDim,
    detailTableTab,
    ciTabKey,
    activeDimKey,
    overviewCards,
    compareCards,
    detailPlat,
    detailCard,
    opDetailRows,
    inferDetailTabRows,
    inferNvidiaTabRows,
    detailTestEnvLine,
    detailTestEnvSourceHint,
    trainDetailRows,
    commDetailRows,
    commNvidiaBaselineRows,
    lineChartOption,
    barChartOption,
    ciChartOption,
    compareScoreOption,
    compareLatencyOption,
    detailTitle,
    tableNotice,
    scoreTabHint,
    comparePageTitle,
    comparePageSubtitle,
    compareKpiBlocks,
    compareTableRows,
    detailKpiCells,
    avgOpScore,
    togglePlat,
    selectAll,
    selectDomestic,
    setDim,
    setFilter,
    toggleSort,
    toggleCompare,
    clearCompare,
    switchMainView,
    opTagKeys,
    setOp,
    setPrec,
    setInferTab,
    parseLatencyMs,
  }
}

export type InfiniDashboardStore = ReturnType<typeof createInfiniDashboardStore>
