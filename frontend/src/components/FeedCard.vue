<template>
  <article class="feed-card" @click="$emit('open', post.post_id)">
    <header class="card-head">
      <span class="board-badge">{{ post.board || '校园论坛' }}</span>
      <span class="time-text">{{ timeText(post.publish_time) }}</span>
    </header>
    <h2>{{ post.title || '无标题帖子' }}</h2>
    <p class="feed-content">{{ post.content || '这个帖子暂时没有正文内容。' }}</p>
    <div class="tag-row">
      <TagBadge v-for="tag in tagList" :key="tag" :label="tag" />
    </div>
    <RecommendInfo :post="post" />
    <footer class="card-foot">
      <span class="author-dot">●</span>
      <span class="author">{{ post.author_id || '匿名同学' }}</span>
      <span>◎ {{ number(post.view_count) }}</span>
      <span>♥ {{ number(post.like_count) }}</span>
      <span>□ {{ number(post.comment_count) }}</span>
    </footer>
  </article>
</template>

<script setup>
import { computed } from 'vue'
import RecommendInfo from './RecommendInfo.vue'
import TagBadge from './TagBadge.vue'

const props = defineProps({
  post: {
    type: Object,
    default: () => ({})
  }
})

defineEmits(['open'])

const tagList = computed(() =>
  String(props.post.tags || '')
    .split('|')
    .map((tag) => tag.trim())
    .filter(Boolean)
    .slice(0, 5)
)

function number(value) {
  const n = Number(value || 0)
  if (n >= 10000) return `${(n / 10000).toFixed(1)}w`
  return n
}

function timeText(value) {
  if (!value) return '刚刚'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return String(value).slice(0, 16)
  const hours = Math.max(0, Math.floor((Date.now() - date.getTime()) / 3600000))
  if (hours < 1) return '刚刚'
  if (hours < 24) return `${hours}小时前`
  return `${Math.floor(hours / 24)}天前`
}
</script>
