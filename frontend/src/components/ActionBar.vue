<template>
  <aside class="action-bar">
    <button class="action-btn like-btn" :class="{ active: post.liked }" @click="$emit('like')">
      <span>♥</span>
      <small>{{ count(post.like_count) }}</small>
    </button>
    <button class="action-btn collect-btn" :class="{ active: post.collected }" @click="$emit('collect')">
      <span>★</span>
      <small>{{ count(post.collect_count) }}</small>
    </button>
    <button class="action-btn comment-btn" @click="$emit('comment')">
      <span class="comment-icon" aria-hidden="true"></span>
      <small>{{ count(post.comment_count) }}</small>
    </button>
    <button class="action-btn skip-btn" @click="$emit('skip')">
      <svg class="broken-heart-icon" viewBox="0 0 32 32" aria-hidden="true">
        <path d="M16 28C9.4 22.1 4 17.2 4 11.6C4 8 6.7 5.5 10 5.5C12 5.5 14 6.5 15.1 8.2L16 9.6L16.9 8.2C18 6.5 20 5.5 22 5.5C25.3 5.5 28 8 28 11.6C28 17.2 22.6 22.1 16 28Z" />
        <path class="crack" d="M17 9L14 14H18L15 20H19L16 28" />
      </svg>
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
