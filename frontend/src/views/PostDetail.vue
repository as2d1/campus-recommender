<template>
  <div class="page detail-page">
    <header class="top-header">
      <button class="back-btn" @click="$router.back()">‹</button>
      <div>
        <h1>帖子详情</h1>
        <p>{{ post.board || '校园论坛' }}</p>
      </div>
    </header>

    <LoadingState v-if="loading" text="正在加载帖子详情..." />
    <EmptyState v-else-if="error" title="帖子加载失败" :description="error" button-text="返回推荐流" @action="$router.push('/feed')" />

    <template v-else>
      <article class="detail-card">
        <span class="board-badge">{{ post.board || '校园论坛' }}</span>
        <h2>{{ post.title || '无标题帖子' }}</h2>
        <p class="detail-content">{{ post.content || '暂无正文' }}</p>
        <div class="tag-row">
          <TagBadge v-for="tag in tagList" :key="tag" :label="tag" />
        </div>
        <div class="detail-meta">
          <span>作者：{{ post.author_id || '匿名同学' }}</span>
          <span>{{ post.publish_time || '未知时间' }}</span>
        </div>
        <div class="detail-stats">
          <span>◎ {{ stats.view_count || 0 }}</span>
          <span>♥ {{ stats.like_count || 0 }}</span>
          <span>□ {{ stats.comment_count || 0 }}</span>
          <span>★ {{ stats.collect_count || 0 }}</span>
        </div>
      </article>

      <section class="comment-section">
        <h3>全部评论（{{ comments.length }}）</h3>
        <article v-for="comment in comments" :key="comment.comment_id" class="comment-card">
          <div class="comment-avatar">●</div>
          <div>
            <b>{{ comment.user_id || '匿名同学' }}</b>
            <p>{{ comment.content || '这位同学什么也没说。' }}</p>
            <small>{{ comment.publish_time || '' }}</small>
          </div>
        </article>
        <EmptyState v-if="comments.length === 0" title="暂无评论" description="成为第一个交流的同学。" />
      </section>

      <div class="detail-actions">
        <button @click="sendAction('like')">点赞</button>
        <button @click="sendAction('collect')">收藏</button>
        <button @click="$router.push('/feed')">返回推荐流</button>
      </div>
      <p v-if="message" class="toast detail-toast">{{ message }}</p>
    </template>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import { useRoute } from 'vue-router'
import EmptyState from '../components/EmptyState.vue'
import LoadingState from '../components/LoadingState.vue'
import TagBadge from '../components/TagBadge.vue'
import { DEFAULT_USER_ID, getPostDetail, recordBehavior } from '../api/request'

const route = useRoute()
const userId = localStorage.getItem('campus_user_id') || DEFAULT_USER_ID
const post = ref({})
const stats = ref({})
const comments = ref([])
const loading = ref(false)
const error = ref('')
const message = ref('')

const tagList = computed(() =>
  String(post.value.tags || '')
    .split('|')
    .map((tag) => tag.trim())
    .filter(Boolean)
    .slice(0, 8)
)

loadDetail()

async function loadDetail() {
  loading.value = true
  error.value = ''
  try {
    const detail = await getPostDetail(route.params.id)
    post.value = detail.post || {}
    stats.value = detail.stats || {}
    comments.value = detail.comments || []
    await recordBehavior({
      user_id: userId,
      post_id: String(route.params.id),
      action_type: 'view',
      dwell_time: 20,
      context: { time_period: '晚上', location: '宿舍区', device_type: 'pc', scene: '详情页' }
    })
  } catch (err) {
    error.value = err.message
  } finally {
    loading.value = false
  }
}

async function sendAction(actionType) {
  try {
    await recordBehavior({
      user_id: userId,
      post_id: String(route.params.id),
      action_type: actionType,
      dwell_time: 30,
      context: { time_period: '晚上', location: '宿舍区', device_type: 'pc', scene: '详情页' }
    })
    message.value = actionType === 'like' ? '已点赞' : '已收藏'
    setTimeout(() => (message.value = ''), 1500)
  } catch (err) {
    message.value = err.message
  }
}
</script>
