import type { CardRow } from '@/features/dashboard/dashboardFilterHelpers'
import {
  inferBatchPillsUnion,
  inferInLenPillsUnion,
  type InferTablePack,
} from '@/features/dashboard/inferBenchmark'
import { trainFrameworkPillsUnion } from '@/features/dashboard/trainBenchmark'
import { bwVsNvidiaPercent, type BwDetailRow } from '@/features/dashboard/bwBenchmark'
import {
  BENCHMARK_DATA_META,
  BW_CARD_FROM_FILES,
  BW_TABLE_FROM_FILES,
  COMM_CARD_FROM_FILES,
  COMM_TABLE_FROM_FILES,
  INFER_CARD_FROM_FILES,
  INFER_TABLE_FROM_FILES,
  OP_CARD_FROM_FILES,
  OP_TABLE_FROM_FILES,
  TRAIN_CARD_FROM_FILES,
  TRAIN_TABLE_FROM_FILES,
} from './generatedFromFiles'

/** 由 `npm run generate:data` 写入，含各维度数据源路径与日期元信息 */
export { BENCHMARK_DATA_META }

/** 自 dashboard_preview.html 同步的静态配置，勿改业务含义 */

/** 用 `new_data/operator`、`new_data/infer`、`new_data/train`、`new_data/comm`、`new_data/bw` 下生成数据覆盖对应平台（npm run generate:data） */
function mergeByPlatform<T extends Record<string, unknown>>(base: T, patch: Partial<T>): T {
  const out = { ...base }
  for (const key of Object.keys(patch)) {
    const v = patch[key as keyof T]
    if (
      v !== undefined &&
      v !== null &&
      typeof v === 'object' &&
      !Array.isArray(v) &&
      Object.keys(v as object).length > 0
    ) {
      ;(out as Record<string, unknown>)[key] = v
    }
  }
  return out
}

/** 训练 / 通信详情：按平台整表替换（值为行数组；空数组表示「无导入数据」不写入） */
function mergeTrainTableByPlatform<T extends Record<string, unknown[]>>(
  base: T,
  patch: Partial<Record<string, unknown[]>>,
): T {
  const out = { ...base } as T
  for (const k of Object.keys(patch)) {
    const v = patch[k as keyof typeof patch]
    if (Array.isArray(v) && v.length > 0) (out as Record<string, unknown[]>)[k] = v as unknown[]
  }
  return out
}

function bwStaticRow(r: {
  model: string
  add: number | null
  copy: number | null
  scale: number | null
  triad: number | null
  avg: number | null
  date?: string
  remarks?: string
  tester?: string
}): BwDetailRow {
  const out: BwDetailRow = {
    model: r.model,
    add: r.add,
    copy: r.copy,
    scale: r.scale,
    triad: r.triad,
    avg: r.avg,
    vsNvidia: r.avg != null ? bwVsNvidiaPercent(r.avg) : 100,
  }
  if (r.date) out.date = r.date
  if (r.remarks) out.remarks = r.remarks
  if (r.tester) out.tester = r.tester
  return out
}

export const PLATFORMS = [
  {key:'nvidia',    name:'NVIDIA A100', type:'国际标杆', color:'#76b900', domestic:false, logo:'NV'},
  {key:'mthreads',  name:'摩尔线程',   type:'国产',     color:'#0066cc', domestic:true,  logo:'MT'},
  {key:'cambricon', name:'寒武纪',      type:'国产',     color:'#c0392b', domestic:true,  logo:'CB'},
  {key:'metax',     name:'沐曦',        type:'国产',     color:'#1a73e8', domestic:true,  logo:'MX'},
  {key:'iluvatar',  name:'天数智芯',    type:'国产',     color:'#e65c00', domestic:true,  logo:'ICX'},
  {key:'ascend',    name:'昇腾',        type:'国产',     color:'#cf0a2c', domestic:true,  logo:'AS'},
  {key:'hygon',     name:'海光',        type:'国产',     color:'#c8a800', domestic:true,  logo:'HG'},
  {key:'generic',   name:'阿里 PPU',   type:'国产',     color:'#ff6a00', domestic:true,  logo:'ALI'},
];

const CARD_DATA_OP_STATIC: CardRow[] = [
  {key:'nvidia',    ownScore:784,  openScore:100, ownVal:'0.0088ms', openVal:'0.0620ms', n:156, extra:'12 算子', ownFw:'InfiniCore ✦', openFw:'PyTorch', adv:true,  advTxt:'自研快 973%'},
  {key:'mthreads',  ownScore:1245, openScore:100, ownVal:'0.0278ms', openVal:'0.1914ms', n:89,  extra:'8 算子',  ownFw:'InfiniCore ✦', openFw:'PyTorch', adv:true,  advTxt:'自研快 1458%'},
  {key:'cambricon', ownScore:1107, openScore:100, ownVal:'0.0170ms', openVal:'0.9489ms', n:134, extra:'10 算子', ownFw:'InfiniCore ✦', openFw:'PyTorch', adv:true,  advTxt:'自研快 1007%'},
  {key:'metax',     ownScore:581,  openScore:100, ownVal:'0.0245ms', openVal:'0.1198ms', n:156, extra:'7 算子',  ownFw:'InfiniCore ✦', openFw:'PyTorch', adv:true,  advTxt:'自研快 481%'},
  {key:'iluvatar',  ownScore:542,  openScore:100, ownVal:'0.0194ms', openVal:'0.0905ms', n:45,  extra:'5 算子',  ownFw:'InfiniCore ✦', openFw:'PyTorch', adv:true,  advTxt:'自研快 442%'},
  {key:'hygon',     ownScore:1013, openScore:100, ownVal:'0.0122ms', openVal:'0.1232ms', n:156, extra:'7 算子',  ownFw:'InfiniCore ✦', openFw:'PyTorch', adv:true,  advTxt:'自研快 913%'},
]

function mergeOpDimensionCards(staticCards: CardRow[]): CardRow[] {
  const patch = OP_CARD_FROM_FILES as Record<string, Partial<CardRow>>
  const keys = new Set(staticCards.map((c) => c.key))
  const merged: CardRow[] = staticCards.map((c) => ({ ...c, ...(patch[c.key] || {}) }))
  for (const key of Object.keys(patch)) {
    if (!keys.has(key)) merged.push(patch[key] as CardRow)
  }
  const order = PLATFORMS.map((p) => p.key)
  merged.sort((a, b) => order.indexOf(a.key) - order.indexOf(b.key))
  return merged
}

function mergeInferDimensionCards(staticCards: CardRow[]): CardRow[] {
  const patch = INFER_CARD_FROM_FILES as Record<string, Partial<CardRow>>
  const keys = new Set(staticCards.map((c) => c.key))
  const merged: CardRow[] = staticCards.map((c) => ({ ...c, ...(patch[c.key] || {}) }))
  for (const key of Object.keys(patch)) {
    if (!keys.has(key)) merged.push(patch[key] as CardRow)
  }
  const order = PLATFORMS.map((p) => p.key)
  merged.sort((a, b) => order.indexOf(a.key) - order.indexOf(b.key))
  return merged
}

function mergeTrainDimensionCards(staticCards: CardRow[]): CardRow[] {
  const patch = TRAIN_CARD_FROM_FILES as Record<string, Partial<CardRow>>
  const keys = new Set(staticCards.map((c) => c.key))
  const merged: CardRow[] = staticCards.map((c) => ({ ...c, ...(patch[c.key] || {}) }))
  for (const key of Object.keys(patch)) {
    if (!keys.has(key)) merged.push(patch[key] as CardRow)
  }
  const order = PLATFORMS.map((p) => p.key)
  merged.sort((a, b) => order.indexOf(a.key) - order.indexOf(b.key))
  return merged
}

const CARD_DATA_INFER_STATIC: CardRow[] = [
  {key:'nvidia',    ownScore:100, openScore:100, ownVal:'13.5K tok/s', openVal:'3.3K tok/s', n:30, extra:'batch=64 in=256', ownFw:'Prefill ✦', openFw:'Decode', adv:true,  advTxt:'基准平台'},
  {key:'mthreads',  ownScore:90,  openScore:52,  ownVal:'7.3K tok/s',  openVal:'1.8K tok/s', n:30, extra:'batch=64 in=32',  ownFw:'Prefill ✦', openFw:'Decode', adv:false, advTxt:'落后 10% vs A100'},
  {key:'cambricon', ownScore:17,  openScore:20,  ownVal:'1.4K tok/s',  openVal:'221 tok/s',  n:30, extra:'batch=16 in=32',  ownFw:'Prefill ✦', openFw:'Decode', adv:false, advTxt:'落后 83% vs A100'},
  {key:'hygon',     ownScore:103, openScore:45,  ownVal:'8.4K tok/s',  openVal:'33.4K tok/s',n:30, extra:'batch=64 in=32',  ownFw:'Prefill ✦', openFw:'Decode', adv:true,  advTxt:'领先 3% vs A100'},
  {key:'metax',     ownScore:82,  openScore:88,  ownVal:'1.4K tok/s',  openVal:'66.8K tok/s',n:30, extra:'batch=1 in=32',   ownFw:'Prefill ✦', openFw:'Decode', adv:false, advTxt:'落后 18% vs A100'},
  {key:'generic',   ownScore:140, openScore:110, ownVal:'2.5K tok/s',  openVal:'83.7K tok/s',n:30, extra:'batch=1 in=32',   ownFw:'Prefill ✦', openFw:'Decode', adv:true,  advTxt:'领先 40% vs A100'},
]

// Card data per dimension — ownFw=自研框架 label, openFw=开源框架 label
const CARD_DATA_TRAIN_STATIC: CardRow[] = [
  {key:'nvidia', ownScore:100, openScore:null, ownVal:'2564 tpps', openVal:null, n:4, extra:'Megatron · llama3-8b · 8 GPU', ownFw:'Megatron', openFw:'', adv:true,  advTxt:'训练基准'},
  {key:'metax',  ownScore:17,  openScore:null, ownVal:'438 tpps',  openVal:null, n:1, extra:'Megatron · llama3-8b · 8 GPU', ownFw:'Megatron', openFw:'', adv:false, advTxt:'落后 83% vs NVIDIA'},
]

function mergeCommDimensionCards(staticCards: CardRow[]): CardRow[] {
  const patch = COMM_CARD_FROM_FILES as Record<string, Partial<CardRow>>
  const keys = new Set(staticCards.map((c) => c.key))
  const merged: CardRow[] = staticCards.map((c) => ({ ...c, ...(patch[c.key] || {}) }))
  for (const key of Object.keys(patch)) {
    if (!keys.has(key)) merged.push(patch[key] as CardRow)
  }
  const order = PLATFORMS.map((p) => p.key)
  merged.sort((a, b) => order.indexOf(a.key) - order.indexOf(b.key))
  return merged
}

const CARD_DATA_COMM_STATIC: CardRow[] = [
  {key:'nvidia', ownScore:100, openScore:100, ownVal:'270 GB/s', openVal:'35 GB/s', n:2, extra:'NVLink',      ownFw:'p2p',      openFw:'allreduce', adv:true,  advTxt:'通信基准'},
  {key:'metax',  ownScore:20,  openScore:131, ownVal:'53.1 GB/s',openVal:'45.8 GB/s',n:2, extra:'MetaxLink',  ownFw:'p2p',      openFw:'allreduce', adv:true,  advTxt:'allreduce 超越基准'},
]

function mergeBwDimensionCards(staticCards: CardRow[]): CardRow[] {
  const patch = BW_CARD_FROM_FILES as Record<string, Partial<CardRow>>
  const keys = new Set(staticCards.map((c) => c.key))
  const merged: CardRow[] = staticCards.map((c) => ({ ...c, ...(patch[c.key] || {}) }))
  for (const key of Object.keys(patch)) {
    if (!keys.has(key)) merged.push(patch[key] as CardRow)
  }
  const order = PLATFORMS.map((p) => p.key)
  merged.sort((a, b) => order.indexOf(a.key) - order.indexOf(b.key))
  return merged
}

const CARD_DATA_BW_STATIC: CardRow[] = [
  {key:'nvidia',    ownScore:null, openScore:null, ownVal:'1607.5 GB/s', openVal:null, n:4, extra:'A100 · add/copy/scale/triad', ownFw:'HBM均值', openFw:'', adv:true,  advTxt:'访存基准'},
  {key:'cambricon', ownScore:null, openScore:null, ownVal:'2131.4 GB/s', openVal:null, n:4, extra:'MLU590', ownFw:'HBM均值', openFw:'', adv:true,  advTxt:'超越 A100'},
  {key:'mthreads',  ownScore:null, openScore:null, ownVal:'1400.9 GB/s', openVal:null, n:4, extra:'S5000', ownFw:'HBM均值', openFw:'', adv:false, advTxt:'落后 A100 13%'},
  {key:'ascend',    ownScore:null, openScore:null, ownVal:'1540.0 GB/s', openVal:null, n:1, extra:'910B3', ownFw:'HBM均值', openFw:'', adv:false, advTxt:'落后 A100 4%'},
  {key:'metax',     ownScore:null, openScore:null, ownVal:'1677.7 GB/s', openVal:null, n:4, extra:'C500', ownFw:'HBM均值', openFw:'', adv:true,  advTxt:'超越 A100 4%'},
  {key:'iluvatar',  ownScore:null, openScore:null, ownVal:'586.0 GB/s',  openVal:null, n:1, extra:'TG150', ownFw:'HBM均值', openFw:'', adv:false, advTxt:'落后 A100 64%'},
]

export const CARD_DATA = {
  op: mergeOpDimensionCards(CARD_DATA_OP_STATIC),
  infer: mergeInferDimensionCards(CARD_DATA_INFER_STATIC),
  train: mergeTrainDimensionCards(CARD_DATA_TRAIN_STATIC),
  comm: mergeCommDimensionCards(CARD_DATA_COMM_STATIC),
  bw: mergeBwDimensionCards(CARD_DATA_BW_STATIC),
}

// Detail ops data — scores displayed in op-tag badges, table updates on switch
const OP_TABLE_STATIC = {
  nvidia:{
    CausalSoftmax:[
      {shape:'M=3, N=3',    dtype:'FP16', ic:0.006581, pt:0.065546},
      {shape:'M=3, N=3',    dtype:'BF16', ic:0.006427, pt:0.065766},
      {shape:'M=3, N=3',    dtype:'FP32', ic:0.006381, pt:0.052724},
      {shape:'M=32, N=512', dtype:'FP16', ic:0.006298, pt:0.066401},
      {shape:'M=32, N=512', dtype:'BF16', ic:0.006380, pt:0.065598},
      {shape:'M=32, N=512', dtype:'FP32', ic:0.006350, pt:0.053560},
      {shape:'bs=32,seq=5,hidden=5',  dtype:'FP16', ic:0.006402, pt:0.067001},
      {shape:'bs=32,seq=5,hidden=5',  dtype:'BF16', ic:0.006422, pt:0.068901},
      {shape:'bs=32,seq=20,hidden=512',dtype:'FP16', ic:0.013301, pt:0.067900},
    ],
    RMSNorm:[
      {shape:'M=1, N=4096',   dtype:'FP16', ic:0.005120, pt:0.046200},
      {shape:'M=1, N=4096',   dtype:'BF16', ic:0.005230, pt:0.046500},
      {shape:'M=32, N=4096',  dtype:'FP16', ic:0.006100, pt:0.053200},
      {shape:'M=32, N=4096',  dtype:'BF16', ic:0.006200, pt:0.053500},
      {shape:'M=32, N=4096',  dtype:'FP32', ic:0.007400, pt:0.058900},
      {shape:'M=128, N=4096', dtype:'FP16', ic:0.007800, pt:0.060100},
    ],
    Embedding:[
      {shape:'vocab=32000,dim=4096', dtype:'FP16', ic:0.006500, pt:0.010400},
      {shape:'vocab=32000,dim=4096', dtype:'BF16', ic:0.006600, pt:0.010500},
      {shape:'vocab=64000,dim=4096', dtype:'FP16', ic:0.006800, pt:0.010800},
    ],
    TopK:[
      {shape:'N=32000,K=1', dtype:'FP16', ic:0.312600, pt:0.009700},
      {shape:'N=32000,K=5', dtype:'FP16', ic:0.318000, pt:0.009800},
      {shape:'N=32000,K=1', dtype:'BF16', ic:0.315000, pt:0.009700},
    ],
    MatMul:[
      {shape:'M=1,N=4096,K=4096',  dtype:'FP16', ic:0.008400, pt:0.008600},
      {shape:'M=1,N=4096,K=4096',  dtype:'BF16', ic:0.008300, pt:0.008500},
      {shape:'M=32,N=4096,K=4096', dtype:'FP16', ic:0.012100, pt:0.014300},
      {shape:'M=32,N=4096,K=4096', dtype:'BF16', ic:0.012000, pt:0.014100},
    ],
    Add:[
      {shape:'N=4096',   dtype:'FP16', ic:0.010500, pt:0.006300},
      {shape:'N=4096',   dtype:'BF16', ic:0.010400, pt:0.006200},
      {shape:'N=16384',  dtype:'FP16', ic:0.010100, pt:0.006300},
      {shape:'N=16384',  dtype:'BF16', ic:0.009961, pt:0.005530},
    ],
    SiLU:[
      {shape:'N=4096',  dtype:'FP16', ic:0.009400, pt:0.011300},
      {shape:'N=4096',  dtype:'BF16', ic:0.009500, pt:0.011400},
      {shape:'N=16384', dtype:'FP16', ic:0.009600, pt:0.011500},
    ],
  },
  mthreads:{
    CausalSoftmax:[
      {shape:'M=3, N=3',    dtype:'FP16', ic:0.008200, pt:0.091000},
      {shape:'M=3, N=3',    dtype:'BF16', ic:0.008100, pt:0.092000},
      {shape:'M=32, N=512', dtype:'FP16', ic:0.007800, pt:0.098000},
      {shape:'M=32, N=512', dtype:'BF16', ic:0.007900, pt:0.097000},
    ],
    RMSNorm:[
      {shape:'M=32, N=4096',  dtype:'FP16', ic:0.017400, pt:0.214400},
      {shape:'M=32, N=4096',  dtype:'BF16', ic:0.017600, pt:0.215000},
      {shape:'M=128, N=4096', dtype:'FP16', ic:0.018200, pt:0.220000},
    ],
    MatMul:[
      {shape:'M=1,N=4096,K=4096',  dtype:'FP16', ic:0.021000, pt:0.028000},
      {shape:'M=32,N=4096,K=4096', dtype:'FP16', ic:0.032000, pt:0.045000},
    ],
    Add:[
      {shape:'N=4096',  dtype:'FP16', ic:0.032000, pt:0.018000},
      {shape:'N=16384', dtype:'FP16', ic:0.031000, pt:0.017500},
    ],
  },
  hygon:{
    CausalSoftmax:[
      {shape:'M=3, N=3',    dtype:'FP16', ic:0.009800, pt:0.120000},
      {shape:'M=3, N=3',    dtype:'BF16', ic:0.009700, pt:0.121000},
      {shape:'M=32, N=512', dtype:'FP16', ic:0.009500, pt:0.125000},
    ],
    RMSNorm:[
      {shape:'M=32, N=4096', dtype:'FP16', ic:0.012200, pt:0.123200},
      {shape:'M=32, N=4096', dtype:'BF16', ic:0.012400, pt:0.124000},
    ],
    Add:[
      {shape:'N=4096',  dtype:'FP16', ic:0.013500, pt:0.009800},
      {shape:'N=16384', dtype:'BF16', ic:0.013200, pt:0.009600},
    ],
  },
  metax:{
    CausalSoftmax:[
      {shape:'M=3, N=3',    dtype:'FP16', ic:0.018500, pt:0.098000},
      {shape:'M=32, N=512', dtype:'FP16', ic:0.018200, pt:0.101000},
    ],
    RMSNorm:[
      {shape:'M=32, N=4096', dtype:'FP16', ic:0.024500, pt:0.119800},
      {shape:'M=32, N=4096', dtype:'BF16', ic:0.024800, pt:0.120500},
    ],
    MatMul:[
      {shape:'M=1,N=4096,K=4096', dtype:'FP16', ic:0.038000, pt:0.042000},
    ],
  },
};

// ── 推理详情数据（静态占位；dtype / remarks / date 与 CSV 字段对齐）
const INFER_TABLE_STATIC = {
  nvidia:{
    prefill:[
      {batch:1,  inLen:32,   outLen:128, model:'9G8B', dtype:'BF16', nGpu:1, remarks:'NVIDIA A100', date:'2026-04-29', tps:3800,  ttft:8.2,  framework:'InfiniLM'},
      {batch:4,  inLen:32,   outLen:128, model:'9G8B', dtype:'BF16', nGpu:1, remarks:'NVIDIA A100', date:'2026-04-29', tps:8200,  ttft:9.1,  framework:'InfiniLM'},
      {batch:16, inLen:256,  outLen:128, model:'9G8B', dtype:'BF16', nGpu:1, remarks:'NVIDIA A100', date:'2026-04-29', tps:11400, ttft:12.3, framework:'InfiniLM'},
      {batch:64, inLen:256,  outLen:128, model:'9G8B', dtype:'BF16', nGpu:1, remarks:'NVIDIA A100', date:'2026-04-29', tps:12900, ttft:24.1, framework:'InfiniLM'},
      {batch:64, inLen:4096, outLen:128, model:'9G8B', dtype:'BF16', nGpu:1, remarks:'NVIDIA A100', date:'2026-04-29', tps:9800,  ttft:89.5, framework:'InfiniLM'},
    ],
    decode:[
      {batch:1,  inLen:32,   outLen:128, model:'9G8B', dtype:'BF16', nGpu:1, remarks:'NVIDIA A100', date:'2026-04-29', tps:1200, framework:'InfiniLM'},
      {batch:4,  inLen:32,   outLen:128, model:'9G8B', dtype:'BF16', nGpu:1, remarks:'NVIDIA A100', date:'2026-04-29', tps:2800, framework:'InfiniLM'},
      {batch:16, inLen:256,  outLen:128, model:'9G8B', dtype:'BF16', nGpu:1, remarks:'NVIDIA A100', date:'2026-04-29', tps:3100, framework:'InfiniLM'},
      {batch:64, inLen:256,  outLen:128, model:'9G8B', dtype:'BF16', nGpu:1, remarks:'NVIDIA A100', date:'2026-04-29', tps:3700, framework:'InfiniLM'},
      {batch:64, inLen:4096, outLen:128, model:'9G8B', dtype:'BF16', nGpu:1, remarks:'NVIDIA A100', date:'2026-04-29', tps:2900, framework:'InfiniLM'},
    ],
  },
  mthreads:{
    prefill:[
      {batch:1,  inLen:32,  outLen:128, model:'9G8B', dtype:'BF16', nGpu:1, remarks:'摩尔 MTT S5000', date:'2026-04-29', tps:1800, ttft:18.4, framework:'InfiniLM'},
      {batch:4,  inLen:32,  outLen:128, model:'9G8B', dtype:'BF16', nGpu:1, remarks:'摩尔 MTT S5000', date:'2026-04-29', tps:4200, ttft:22.1, framework:'InfiniLM'},
      {batch:64, inLen:32,  outLen:128, model:'9G8B', dtype:'BF16', nGpu:1, remarks:'摩尔 MTT S5000', date:'2026-04-29', tps:7300, ttft:38.6, framework:'InfiniLM'},
    ],
    decode:[
      {batch:1,  inLen:32,  outLen:128, model:'9G8B', dtype:'BF16', nGpu:1, remarks:'摩尔 MTT S5000', date:'2026-04-29', tps:580,  framework:'InfiniLM'},
      {batch:4,  inLen:32,  outLen:128, model:'9G8B', dtype:'BF16', nGpu:1, remarks:'摩尔 MTT S5000', date:'2026-04-29', tps:1200, framework:'InfiniLM'},
      {batch:64, inLen:32,  outLen:128, model:'9G8B', dtype:'BF16', nGpu:1, remarks:'摩尔 MTT S5000', date:'2026-04-29', tps:1800, framework:'InfiniLM'},
    ],
  },
};

export const OP_TABLE = mergeByPlatform(
  OP_TABLE_STATIC as unknown as Record<string, unknown>,
  OP_TABLE_FROM_FILES as unknown as Partial<Record<string, unknown>>,
) as typeof OP_TABLE_STATIC

/**
 * 算子类型 pills（并集）：与 CSV `op_name` 经 `generate:data` → OP_TABLE 键一致（`opNameToKey`）。
 * 用于概览 / 对比顶栏及 `filterState` 索引基准；详情页算子类型 UI 在 `DashboardFilterBar` 按当前平台再筛子集。
 */
function operatorTypePillsFromOpTable(tbl: typeof OP_TABLE): string[] {
  const keys = new Set<string>()
  for (const platOps of Object.values(tbl)) {
    if (!platOps || typeof platOps !== 'object') continue
    for (const opKey of Object.keys(platOps)) keys.add(opKey)
  }
  return [...keys].sort((a, b) => a.localeCompare(b, undefined, { sensitivity: 'base' }))
}

/** 与 CSV `dtype` / 行内 `dtype` 一致：trim + 大写，用于 pills 与行比较 */
function normalizeOpDtypeLabel(v: unknown): string {
  return String(v ?? '').trim().toUpperCase()
}

/** 某平台算子表中出现过的精度（dtype）集合 */
export function dtypesForOpPlatform(tbl: typeof OP_TABLE, platKey: string): Set<string> {
  const set = new Set<string>()
  const rowTbl = tbl as Record<string, Record<string, { dtype?: string }[] | undefined> | undefined>
  const platOps = rowTbl[platKey]
  if (!platOps) return set
  for (const rows of Object.values(platOps)) {
    if (!Array.isArray(rows)) continue
    for (const row of rows) {
      const d = normalizeOpDtypeLabel(row?.dtype)
      if (d) set.add(d)
    }
  }
  return set
}

/**
 * 精度（dtype）pills（并集）：OP_TABLE 各行 `dtype` 去重，与 CSV 一致。
 * 概览 / 对比用 DIMS 全量；详情顶栏在 `DashboardFilterBar` 按当前平台再筛子集。
 */
function dtypePrecPillsUnionFromOpTable(tbl: typeof OP_TABLE): string[] {
  const keys = new Set<string>()
  const rowTbl = tbl as Record<string, Record<string, { dtype?: string }[] | undefined>>
  for (const pk of Object.keys(rowTbl)) {
    for (const d of dtypesForOpPlatform(tbl, pk)) keys.add(d)
  }
  return [...keys].sort((a, b) => a.localeCompare(b, undefined, { sensitivity: 'base' }))
}

export const INFER_TABLE = mergeByPlatform(
  INFER_TABLE_STATIC as unknown as Record<string, unknown>,
  INFER_TABLE_FROM_FILES as unknown as Partial<Record<string, unknown>>,
) as typeof INFER_TABLE_STATIC

const INFER_FOR_DIMS = INFER_TABLE as Record<string, InferTablePack | undefined>

// ── 训练详情数据（`new_data/train/{platform}_train_{date}.xlsx` 经 generate:data 覆盖同 key 平台）
export interface TrainDetailRow {
  framework: string
  model: string
  parallel: string
  dtype: string
  flashAttn: string
  tps: number
  baseline: number
  vsA100: number
  note: string
  nGpu?: number
  seqLen?: number
  microBatchSize?: number
  date?: string
}

const TRAIN_TABLE_STATIC: Record<string, TrainDetailRow[]> = {
  nvidia: [
    {
      framework: 'megatron',
      model: 'llama3-8b',
      parallel: '8 GPU · seq 8192',
      dtype: 'BF16',
      flashAttn: 'on',
      tps: 2564,
      baseline: 2564,
      vsA100: 100,
      note: '',
      nGpu: 8,
      seqLen: 8192,
      microBatchSize: 128,
      date: '2026-04-29',
    },
    {
      framework: 'megatron',
      model: 'llama3-8b',
      parallel: '8 GPU · seq 4096',
      dtype: 'BF16',
      flashAttn: 'on',
      tps: 2890,
      baseline: 2890,
      vsA100: 100,
      note: '',
      nGpu: 8,
      seqLen: 4096,
      date: '2026-04-29',
    },
    {
      framework: 'megatron',
      model: 'llama3-70b',
      parallel: '8 GPU · seq 8192',
      dtype: 'BF16',
      flashAttn: 'on',
      tps: 312,
      baseline: 312,
      vsA100: 100,
      note: 'pipeline=4',
      nGpu: 8,
      seqLen: 8192,
      date: '2026-04-29',
    },
    {
      framework: 'bmtrain',
      model: 'llama3-8b',
      parallel: '8 GPU · seq 2048',
      dtype: 'BF16',
      flashAttn: 'off',
      tps: 1820,
      baseline: 1820,
      vsA100: 100,
      note: '',
      nGpu: 8,
      seqLen: 2048,
      date: '2026-04-29',
    },
  ],
  metax: [
    {
      framework: 'megatron',
      model: 'llama3-8b',
      parallel: '8 GPU · seq 8192',
      dtype: 'BF16',
      flashAttn: 'on',
      tps: 438,
      baseline: 2564,
      vsA100: 17,
      note: '',
      nGpu: 8,
      seqLen: 8192,
      date: '2026-04-29',
    },
  ],
}

export const TRAIN_TABLE = mergeTrainTableByPlatform(
  TRAIN_TABLE_STATIC,
  TRAIN_TABLE_FROM_FILES as unknown as Partial<Record<string, TrainDetailRow[]>>,
) as Record<string, TrainDetailRow[]>

const TRAIN_FOR_DIMS = TRAIN_TABLE as Record<string, { framework: string }[] | undefined>

/** 测试维度与顶栏筛选（推理 Batch / In-len 来自 INFER_TABLE 并集；训练框架 pills 来自 TRAIN_TABLE 并集） */
export const DIMS = [
  {
    key: 'op',
    label: '算子',
    filters: [
      { label: '算子类型', pills: ['全部', ...operatorTypePillsFromOpTable(OP_TABLE)] },
      { label: '精度', pills: ['全部', ...dtypePrecPillsUnionFromOpTable(OP_TABLE)] },
    ],
  },
  {
    key: 'infer',
    label: '推理',
    filters: [
      { label: 'Batch', pills: ['全部', ...inferBatchPillsUnion(INFER_FOR_DIMS)] },
      { label: 'In-len', pills: ['全部', ...inferInLenPillsUnion(INFER_FOR_DIMS)] },
    ],
  },
  {
    key: 'train',
    label: '训练',
    filters: [{ label: '框架', pills: ['全部', ...trainFrameworkPillsUnion(TRAIN_FOR_DIMS)] }],
  },
  {
    key: 'comm',
    label: '通信',
    filters: [{ label: '通信类型', pills: ['全部', 'p2p', 'allreduce'] }],
  },
  {
    key: 'bw',
    label: '访存',
    filters: [{ label: '模式', pills: ['全部', 'add', 'copy', 'scale', 'triad'] }],
  },
]

// ── 通信详情数据（`new_data/comm/{platform}_comm_{date}.xlsx` 经 generate:data 覆盖同 key 平台）
export interface CommDetailRow {
  linkType: string
  commType: string
  nGpu: number
  bw: number
  baseline: number
  vsA100: number
  /** 文件 `remarks`，详情表「备注」列 */
  note: string
  date?: string
}

const COMM_TABLE_STATIC: Record<string, CommDetailRow[]> = {
  nvidia:[
    {linkType:'NVLink',  commType:'p2p',      nGpu:2, bw:270.0, baseline:270.0, vsA100:100, note:'NVLink 单向基准', date:'2026-04-29'},
    {linkType:'NVLink',  commType:'allreduce', nGpu:8, bw:35.0,  baseline:35.0,  vsA100:100, note:'NVLink 单向基准', date:'2026-04-29'},
  ],
  metax:[
    {linkType:'MetaxLink', commType:'p2p',      nGpu:2, bw:53.1, baseline:270.0, vsA100:20,  note:'', date:'2026-04-29'},
    {linkType:'MetaxLink', commType:'allreduce', nGpu:8, bw:45.8, baseline:35.0, vsA100:131, note:'超越 NVLink allreduce', date:'2026-04-29'},
  ],
}

export const COMM_TABLE = mergeTrainTableByPlatform(
  COMM_TABLE_STATIC,
  COMM_TABLE_FROM_FILES as unknown as Partial<Record<string, CommDetailRow[]>>,
) as Record<string, CommDetailRow[]>

// ── 访存详情数据（`new_data/bw/{platform}_bw_{date}.csv` 与 `new_data/bw/bw_template.csv` 经 generate:data 覆盖同 key 平台）
const BW_TABLE_STATIC: Record<string, BwDetailRow[]> = {
  nvidia: [
    bwStaticRow({
      model: 'A100',
      add: 1630.7558,
      copy: 1585.1489,
      scale: 1579.7755,
      triad: 1634.1444,
      avg: 1607.4561,
      date: '2026-04-29',
    }),
  ],
  cambricon: [
    bwStaticRow({
      model: 'MLU590',
      add: 2145.08,
      copy: 2110.65,
      scale: 2116.64,
      triad: 2153.35,
      avg: 2131.43,
      date: '2026-04-29',
    }),
    bwStaticRow({
      model: 'MLU370',
      add: 256.52,
      copy: 261.18,
      scale: 260.33,
      triad: 256.63,
      avg: 258.67,
      date: '2026-04-29',
    }),
  ],
  ascend: [
    bwStaticRow({
      model: '910B3',
      add: 1540,
      copy: 1540,
      scale: 1540,
      triad: 1540,
      avg: 1540,
    }),
  ],
  metax: [
    bwStaticRow({
      model: 'C500',
      add: 1677.67,
      copy: 1677.67,
      scale: 1677.67,
      triad: 1677.67,
      avg: 1677.67,
      date: '2026-04-29',
    }),
  ],
  mthreads: [
    bwStaticRow({
      model: 'S5000',
      add: 1400.9,
      copy: 1400.9,
      scale: 1400.9,
      triad: 1400.9,
      avg: 1400.9,
      date: '2026-04-29',
    }),
  ],
  iluvatar: [
    bwStaticRow({
      model: 'TG150',
      add: 586,
      copy: 586,
      scale: 586,
      triad: 586,
      avg: 586,
      date: '2026-04-29',
    }),
    bwStaticRow({
      model: 'TG200',
      add: null,
      copy: null,
      scale: null,
      triad: null,
      avg: null,
      date: '2026-04-29',
    }),
  ],
}

export const BW_TABLE = mergeTrainTableByPlatform(
  BW_TABLE_STATIC,
  BW_TABLE_FROM_FILES as unknown as Partial<Record<string, BwDetailRow[]>>,
) as Record<string, BwDetailRow[]>

/** 侧栏「数据更新于」：取算子 / 推理 / 训练 / 通信 / 访存 数据源日期的较新者 */
export const DATA_UPDATED_AT = (() => {
  const m = BENCHMARK_DATA_META as {
    opDatasetUpdatedAt?: string
    inferDatasetUpdatedAt?: string
    trainDatasetUpdatedAt?: string
    commDatasetUpdatedAt?: string
    bwDatasetUpdatedAt?: string
  }
  const dates = [
    m?.opDatasetUpdatedAt,
    m?.inferDatasetUpdatedAt,
    m?.trainDatasetUpdatedAt,
    m?.commDatasetUpdatedAt,
    m?.bwDatasetUpdatedAt,
  ].filter(Boolean) as string[]
  if (!dates.length) return '2026-04-29'
  return dates.reduce((a, b) => (a >= b ? a : b))
})()

/**
 * 详情页 CI「运行统计」四格 KPI（接入真实 CI API 前为静态占位，与 HTML 预览一致）
 */
export const CI_SUMMARY = {
  runCount: 46,
  avgSuccessRate: '71.7%',
  last10SuccessRate: '90.0%',
  failureCount: 13,
} as const

/** 详情页 CI 折线图横轴标签（与 dashboard_preview.html Chart.js 一致） */
export const CI_CHART_LABELS = ['4/22', '4/23', '4/24', '4/25', '4/26', '4/27', '4/28', '4/29'] as const

/** 各维度详情页 CI 折线图纵轴序列（按维度切换；与 HTML 各维度 CI 图一致） */
export const CI_SERIES_BY_DIM: Record<string, readonly number[]> = {
  op: [8, 5, 12, 7, 9, 11, 6, 8],
  infer: [5, 8, 6, 9, 7, 10, 8, 9],
  train: [3, 4, 2, 5, 4, 3, 6, 4],
  comm: [2, 3, 2, 4, 3, 2, 3, 2],
  bw: [4, 3, 5, 4, 6, 4, 5, 4],
}

export function getCiSeriesForDim(dimKey: string): number[] {
  const s = CI_SERIES_BY_DIM[dimKey] ?? CI_SERIES_BY_DIM.op
  return [...s]
}
