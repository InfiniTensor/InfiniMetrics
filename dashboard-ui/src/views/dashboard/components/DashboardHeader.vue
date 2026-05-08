<script setup lang="ts">
import { LOGO_INFINITENSOR, LOGO_JIUYUAN } from '@/data'
</script>

<template>
  <!-- 顶栏：品牌与标题（与 HTML header 一致） -->
  <div class="header">
    <div class="logo-wrap">
      <img class="header-logo" :src="LOGO_JIUYUAN" alt="九源 logo" />
      <span class="logo-divider" />
      <img class="header-logo header-logo--sm" :src="LOGO_INFINITENSOR" alt="InfiniTensor logo" />
      <div>
        <div class="logo-title">InfiniBench 国产算力性能基准平台</div>
        <div class="logo-sub">国产 AI 算力 · 统一评测标准 · 持续跟踪优化进展</div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.header-logo {
  height: 44px;
  width: auto;
  object-fit: contain;
  border-radius: 8px;
}
.header-logo--sm {
  height: 38px;
  border-radius: 6px;
  background: #fff;
  padding: 2px;
}
.logo-divider {
  width: 1px;
  height: 32px;
  background: #e0e0e0;
  border-radius: 1px;
}
</style>
