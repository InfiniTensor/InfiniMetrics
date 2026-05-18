import { nextTick, onMounted, onUnmounted, watch, type Ref } from 'vue'

/** vue-echarts 实例暴露的 resize（与 autoresize 叠加，覆盖侧栏伸缩、grid 列宽变化等） */
export type EchartsResizeTarget = { resize?: () => void } | null | undefined

function collectCharts(getCharts: () => EchartsResizeTarget[]): EchartsResizeTarget[] {
  return getCharts().filter(Boolean)
}

/** 对一组图表执行 resize（双 rAF 等布局稳定后再绘） */
export function resizeEchartsBatch(getCharts: () => EchartsResizeTarget[]) {
  void nextTick(() => {
    const charts = collectCharts(getCharts)
    for (const c of charts) c?.resize?.()
    requestAnimationFrame(() => {
      for (const c of collectCharts(getCharts)) c?.resize?.()
    })
  })
}

/**
 * 监听图表区容器尺寸与窗口 resize，使折线/柱图随页面拉伸、缩窄重算宽高。
 * 与 v-chart 的 autoresize 并用：autoresize 盯单图根节点，本 hook 覆盖父级 grid / flex 变化。
 */
export function useEchartsGridResize(
  gridRef: Ref<HTMLElement | null | undefined>,
  getCharts: () => EchartsResizeTarget[],
  watchSources?: () => unknown[],
) {
  let rafId = 0
  const schedule = () => {
    cancelAnimationFrame(rafId)
    rafId = requestAnimationFrame(() => resizeEchartsBatch(getCharts))
  }

  let ro: ResizeObserver | undefined

  onMounted(() => {
    ro = new ResizeObserver(schedule)
    if (gridRef.value) ro.observe(gridRef.value)
    window.addEventListener('resize', schedule, { passive: true })
    schedule()
  })

  onUnmounted(() => {
    ro?.disconnect()
    window.removeEventListener('resize', schedule)
    cancelAnimationFrame(rafId)
  })

  watch(
    gridRef,
    (el, prev) => {
      if (!ro) return
      if (prev) ro.unobserve(prev)
      if (el) {
        ro.observe(el)
        schedule()
      }
    },
    { flush: 'post' },
  )

  if (watchSources) {
    watch(watchSources, schedule, { flush: 'post', deep: true })
  }

  return { scheduleResize: schedule }
}
