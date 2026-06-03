<template>
  <div class="recommend-info">
    <div class="info-title">推荐理由</div>
    <p>{{ post.recommend_reason || '综合你的兴趣和帖子质量进行推荐。' }}</p>
    <div class="score-row">
      <span>召回来源：{{ post.recall_sources || 'unknown' }}</span>
      <b>{{ sourceLabel }}</b>
    </div>
    <div class="score-row">
      <span>DeepFM：{{ formatScore(post.rank_score) }}</span>
      <span>重排分数：{{ formatScore(post.rerank_score) }}</span>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  post: {
    type: Object,
    default: () => ({})
  }
})

const sourceLabel = computed(() => (props.post.rank_score_source === 'deepfm' ? 'DeepFM排序' : '召回降级排序'))

function formatScore(value) {
  const number = Number(value)
  return Number.isFinite(number) ? number.toFixed(4) : '0.0000'
}
</script>
