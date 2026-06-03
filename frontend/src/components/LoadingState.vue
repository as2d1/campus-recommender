<template>
  <div class="state-box">
    <div class="loader"></div>
    <p>{{ text }}</p>
  </div>
</template>

<script setup>
defineProps({
  text: { type: String, default: '正在加载校园推荐...' }
})
</script>
