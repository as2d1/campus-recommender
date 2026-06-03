<template>
  <aside class="action-bar">
    <button class="action-btn" @click="$emit('like')">
      <span>♥</span>
      <small>{{ count(post.like_count) }}</small>
    </button>
    <button class="action-btn" @click="$emit('collect')">
      <span>★</span>
      <small>{{ count(post.collect_count) }}</small>
    </button>
    <button class="action-btn" @click="$emit('comment')">
      <span>●</span>
      <small>{{ count(post.comment_count) }}</small>
    </button>
    <button class="action-btn" @click="$emit('skip')">
      <span>»</span>
      <small>跳过</small>
    </button>
    <button class="action-btn detail-btn" @click="$emit('detail')">
      <span>↗</span>
      <small>详情</small>
    </button>
  </aside>
</template>

<script setup>
defineProps({
  post: {
    type: Object,
    default: () => ({})
  }
})

defineEmits(['like', 'collect', 'comment', 'skip', 'detail'])

function count(value) {
  const number = Number(value || 0)
  if (number >= 10000) return `${(number / 10000).toFixed(1)}w`
  return number
}
</script>
