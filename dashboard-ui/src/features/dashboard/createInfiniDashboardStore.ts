import { Modal } from 'ant-design-vue'
import { computed, ref } from 'vue'
import {
  BW_TABLE,
  CARD_DATA,
  COMM_TABLE,
  DIMS,
  getCiSeriesForDim,
  INFER_TABLE,
  OP_TABLE,
  PLATFORMS,
  TRAIN_TABLE,
} from '@/data'
import {
  applyCardFilter,
  avgOpScore,
  parseLatencyMs,
  type CardRow,
} from '@/features/dashboard/dashboardFilterHelpers'
import {
  buildBwBarAvg,
  buildBwBarModes,
  buildCiLineOption,
  buildCommBarBw,
  buildCommBarVs,
  buildCompareLatencyBar,
  buildCompareScoreBar,
  buildInferDecodeBarOption,
  buildInferPrefillBarOption,
  buildOpBarAvgOption,
  buildOpLineOption,
  buildTrainBarThroughput,
  buildTrainBarVs,
} from '@/utils/echartsInfini'

export type MainView = 'overview' | 'detail' | 'compare'

const OTABLE = OP_TABLE as Record<string, Record<string, { shape: string; dtype: string; ic: number; pt: number }[]>>

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
    const sorted = [...data].sort((a, b) =>
      sortDesc.value
        ? (b.ownScore ?? 0) - (a.ownScore ?? 0)
        : (a.ownScore ?? 0) - (b.ownScore ?? 0),
    )
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

  const lineChartOption = computed(() => {
    const plat = detailPlat.value
    const dk = activeDimKey.value
    if (dk === 'op') {
      const rows = opDetailRows.value
      if (!rows.length) return {}
      return buildOpLineOption(rows, plat.color)
    }
    if (dk === 'infer') {
      const data = INFER_TABLE[detailState.value.platKey as keyof typeof INFER_TABLE] as
        | { prefill?: { batch: number; inLen: number; tps: number }[]; decode?: { batch: number; inLen: number; tps: number }[] }
        | undefined
      const nv = INFER_TABLE.nvidia
      return buildInferPrefillBarOption(data?.prefill || [], nv?.prefill || [], plat.color)
    }
    if (dk === 'train') {
      const rows = TRAIN_TABLE[detailState.value.platKey as keyof typeof TRAIN_TABLE] as typeof TRAIN_TABLE.nvidia | undefined
      if (!rows?.length) return {}
      return buildTrainBarThroughput(rows, plat.color)
    }
    if (dk === 'comm') {
      const rows = COMM_TABLE[detailState.value.platKey as keyof typeof COMM_TABLE] as typeof COMM_TABLE.nvidia | undefined
      if (!rows?.length) return {}
      return buildCommBarBw(rows, plat.color)
    }
    if (dk === 'bw') {
      const rows = BW_TABLE[detailState.value.platKey as keyof typeof BW_TABLE] as typeof BW_TABLE.nvidia | undefined
      const nvidiaAvg = (BW_TABLE.nvidia?.[0]?.avg ?? 1607.46) as number
      const platColor = plat.color
      const filtered = (rows || []).filter((r) => r.avg != null)
      if (!filtered.length) return {}
      return buildBwBarAvg(filtered, platColor, nvidiaAvg)
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
        const valid = rs.filter((r) => r.ic && r.pt)
        return valid.length
          ? Math.round(valid.reduce((a, r) => a + (r.pt / r.ic) * 100, 0) / valid.length)
          : 0
      })
      return buildOpBarAvgOption(opKeys, scores, plat.color)
    }
    if (dk === 'infer') {
      const data = INFER_TABLE[detailState.value.platKey as keyof typeof INFER_TABLE] as
        | { prefill?: { batch: number; inLen: number; tps: number }[]; decode?: { batch: number; inLen: number; tps: number }[] }
        | undefined
      const nv = INFER_TABLE.nvidia
      return buildInferDecodeBarOption(data?.decode || [], nv?.decode || [], plat.color)
    }
    if (dk === 'train') {
      const rows = TRAIN_TABLE[detailState.value.platKey as keyof typeof TRAIN_TABLE] as typeof TRAIN_TABLE.nvidia | undefined
      if (!rows?.length) return {}
      return buildTrainBarVs(rows)
    }
    if (dk === 'comm') {
      const rows = COMM_TABLE[detailState.value.platKey as keyof typeof COMM_TABLE] as typeof COMM_TABLE.nvidia | undefined
      if (!rows?.length) return {}
      return buildCommBarVs(rows)
    }
    if (dk === 'bw') {
      const rows = BW_TABLE[detailState.value.platKey as keyof typeof BW_TABLE] as typeof BW_TABLE.nvidia | undefined
      const nvidiaRow = BW_TABLE.nvidia?.[0]
      const valid = (rows || []).filter((r) => r.avg != null)
      const bestRow = valid.length ? valid[0] : null
      if (!bestRow || !nvidiaRow) return {}
      return buildBwBarModes(bestRow, nvidiaRow, plat.color)
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
      return '每行得分 = PyTorch延迟 ÷ InfiniCore延迟 × 100（同 shape + dtype）· >100 自研更快 · 平均得分 = 所有行均值 · ✦ InfiniCore 为自研框架'
    if (k === 'infer') return '吞吐量（tokens/s）· 数据越高越好 · vs NVIDIA InfiniLM 同配置'
    if (k === 'train') return '训练吞吐 tpps = tokens per process per second · vs NVIDIA Megatron 基线'
    if (k === 'comm') return '带宽 vs NVIDIA NVLink 基线 · 单向带宽'
    if (k === 'bw')
      return '访存带宽（GB/s）vs NVIDIA 访存带宽 · add / copy / scale / triad 四模式均值'
    return ''
  })

  const scoreTabHint = computed(() => {
    const k = activeDimKey.value
    const msgs: Record<string, string> = {
      op: '每行得分 = PyTorch延迟 ÷ InfiniCore延迟 × 100（同 shape + dtype）· >100 自研更快 · 平均得分 = 所有行均值 · 完整计分体系待定',
      infer: '吞吐量（tokens/s）· 数据越高越好 · vs NVIDIA InfiniLM 同配置',
      train: '训练吞吐 tpps = tokens per process per second · vs NVIDIA Megatron 基线',
      comm: '带宽 vs NVIDIA NVLink 基线 · 单向带宽',
      bw: '访存带宽（GB/s）vs NVIDIA 访存带宽 · add / copy / scale / triad 四模式均值',
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
        { val: c.ownScore ?? '—', lbl: '平均得分', sub: 'InfiniCore vs PyTorch' },
        { val: c.n ?? '—', lbl: '测试记录', sub: c.extra || '' },
        { val: c.ownVal ?? '—', lbl: 'InfiniCore 代表延迟', sub: '自研框架', valSm: true },
        { val: c.openVal ?? '—', lbl: 'PyTorch 代表延迟', sub: '开源基准', valSm: true, muted: true },
      ]
    }
    if (dimKey === 'infer') {
      const inferRows = INFER_TABLE[platKey as keyof typeof INFER_TABLE] as
        | {
            prefill?: { tps: number; ttft?: number; model?: string }[]
            decode?: { tps: number }[]
          }
        | undefined
      const prefill = inferRows?.prefill || []
      const decode = inferRows?.decode || []
      const peakPrefill = prefill.length ? Math.max(...prefill.map((r) => r.tps)) : null
      const peakDecode = decode.length ? Math.max(...decode.map((r) => r.tps)) : null
      const minTTFT = prefill.filter((r) => r.ttft).length
        ? Math.min(...prefill.filter((r) => r.ttft).map((r) => r.ttft!))
        : null
      const model = prefill[0]?.model || c.extra || ''
      return [
        {
          val: peakPrefill != null ? peakPrefill.toLocaleString() : c.ownVal || '—',
          lbl: 'Prefill 峰值 TPS',
          sub: `InfiniLM · ${model}`,
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
          sub: '首 token 延迟',
          valSm: true,
        },
        {
          val: (c.n || prefill.length + decode.length || '—') as string | number,
          lbl: '测试条数',
          sub: 'Prefill + Decode',
        },
      ]
    }
    if (dimKey === 'train') {
      const rows = TRAIN_TABLE[platKey as keyof typeof TRAIN_TABLE] as typeof TRAIN_TABLE.nvidia | undefined
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
          lbl: 'vs A100 基线',
          sub: 'Megatron 同配置',
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
      const rows = COMM_TABLE[platKey as keyof typeof COMM_TABLE] as typeof COMM_TABLE.nvidia | undefined
      const p2p = rows?.find((r) => r.commType === 'p2p')
      const ar = rows?.find((r) => r.commType === 'allreduce')
      return [
        {
          val: p2p ? p2p.bw + ' GB/s' : '—',
          lbl: 'P2P 带宽',
          sub: p2p ? p2p.linkType + ' · ' + p2p.nGpu + 'GPU' : '',
          valSm: true,
        },
        {
          val: ar ? ar.bw + ' GB/s' : '—',
          lbl: 'AllReduce 带宽',
          sub: ar ? ar.linkType + ' · ' + ar.nGpu + 'GPU' : '',
          valSm: true,
        },
        {
          val: p2p ? p2p.vsA100 + '%' : '—',
          lbl: 'P2P vs A100',
          sub: 'NVLink 基线对比',
          valSm: true,
        },
        {
          val: ar ? ar.vsA100 + '%' : '—',
          lbl: 'AllReduce vs A100',
          sub: 'NVLink 基线对比',
          valSm: true,
        },
      ]
    }
    if (dimKey === 'bw') {
      const rows = BW_TABLE[platKey as keyof typeof BW_TABLE] as typeof BW_TABLE.nvidia | undefined
      const best = rows?.length ? rows.reduce((a, b) => ((a.avg ?? 0) > (b.avg ?? 0) ? a : b)) : undefined
      const nvidiaAvg = BW_TABLE.nvidia?.[0]?.avg ?? 1607.46
      const vsA100 =
        best?.avg != null ? Math.round((best.avg / nvidiaAvg) * 100) : null
      return [
        {
          val: best?.avg != null ? best.avg.toFixed(1) + ' GB/s' : '—',
          lbl: 'HBM 均值带宽',
          sub: `${best?.model || ''} · 4模式均值`,
          valSm: true,
        },
        {
          val: vsA100 != null ? vsA100 + '%' : '—',
          lbl: 'vs A100',
          sub: 'A100 基线 1607.5 GB/s',
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

  function openComparePage() {
    const cards = compareCards.value
    if (cards.length < 2) {
      Modal.warning({ title: '提示', content: '请至少选择 2 个有当前维度数据的平台' })
      return
    }
    switchMainView('compare')
  }

  function openDetail(platKey: string) {
    detailState.value.platKey = platKey
    detailState.value.opKey = 'CausalSoftmax'
    detailState.value.prec = '全部'
    detailState.value.inferTab = 'prefill'
    detailTableTab.value = 'data'
    const plat = PLATFORMS.find((p) => p.key === platKey)!
    const dim = DIMS[activeDim.value]
    bcBrand.value = plat.name
    bcDim.value = dim.label
    switchMainView('detail')
  }

  function goBack() {
    switchMainView('overview')
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
    openComparePage,
    openDetail,
    goBack,
    switchMainView,
    opTagKeys,
    setOp,
    setPrec,
    setInferTab,
    parseLatencyMs,
  }
}

export type InfiniDashboardStore = ReturnType<typeof createInfiniDashboardStore>
