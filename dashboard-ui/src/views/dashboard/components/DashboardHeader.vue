<script setup lang="ts">
/** 顶栏使用整图 banner（public/header.png），文案见图片内与 alt */
const HEADER_IMG = '/header.png'
</script>

<template>
  <header class="header" aria-label="InfiniBench 顶栏">
    <div class="header__banner">
      <img
        class="header__img"
        :src="HEADER_IMG"
        width="1600"
        height="200"
        decoding="async"
        fetchpriority="high"
        alt="九源 JIUYUAN · InfiniTensor · InfiniBench 国产算力性能基准平台。国产 AI 算力 · 统一评测标准 · 持续跟踪优化进展"
      />
    </div>
  </header>
</template>

<style scoped>
.header__banner {
  width: 100%;
  min-width: 0;
  padding: 0;
}
.header__img {
  width: 100%;
  max-width: 100%;
  height: auto;
  display: block;
  object-fit: contain;
  object-position: left center;
  vertical-align: top;
}
</style>
