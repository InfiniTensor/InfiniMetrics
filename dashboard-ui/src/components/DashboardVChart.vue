<script setup lang="ts">
import { nextTick, ref, watch } from 'vue'
import VChart from 'vue-echarts'

const props = withDefaults(
  defineProps<{
    option: object
    updateOptions?: object
    /** 与全局 detail 图一致：切换维度时 notMerge，避免轴样式残留 */
    notMerge?: boolean
  }>(),
  { notMerge: false },
)

const chartRef = ref<InstanceType<typeof VChart> | null>(null)

const updateOptions = () =>
  props.updateOptions ?? (props.notMerge ? { notMerge: true } : undefined)

function resize() {
  chartRef.value?.resize()
}

defineExpose({ resize })

watch(
  () => props.option,
  () => {
    void nextTick(() => {
      requestAnimationFrame(resize)
    })
  },
  { deep: true },
)
</script>

<template>
  <v-chart
    ref="chartRef"
    :option="option"
    :update-options="updateOptions"
    autoresize
    v-bind="$attrs"
  />
</template>
