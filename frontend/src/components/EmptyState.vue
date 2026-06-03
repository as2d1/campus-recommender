<template>
  <div class="state-box empty-box">
    <div class="planet">✦</div>
    <h3>{{ title }}</h3>
    <p>{{ description }}</p>
    <button v-if="buttonText" class="primary-btn" @click="$emit('action')">{{ buttonText }}</button>
  </div>
</template>

<script setup>
defineProps({
  title: { type: String, default: '暂无内容' },
  description: { type: String, default: '稍后再来看看吧。' },
  buttonText: { type: String, default: '' }
})

defineEmits(['action'])
</script>
