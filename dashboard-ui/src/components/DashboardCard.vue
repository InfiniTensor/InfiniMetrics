<script setup lang="ts">
withDefaults(
  defineProps<{
    /** 根元素标签，默认 div */
    tag?: string
  }>(),
  {
    tag: 'div',
  },
)
</script>

<template>
  <component :is="tag" class="dashboard-card">
    <slot />
  </component>
</template>

<style scoped>
.dashboard-card {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
  height: 100%;
  overflow: hidden;
  padding: 16px;
  border-radius: 4px;
  background: #fff;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
}
</style>
