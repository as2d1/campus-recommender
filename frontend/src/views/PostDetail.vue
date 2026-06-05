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
        <div v-if="imageUrls.length" class="post-image-grid detail">
          <img
            v-for="url in imageUrls"
            :key="url"
            :src="url"
            alt=""
            loading="lazy"
            @click.stop="openImage(url)"
            @error="hideBrokenImage"
          />
        </div>
        <div class="tag-row">
          <TagBadge v-for="tag in tagList" :key="tag" :label="tag" />
        </div>
        <div class="detail-meta">
          <span>作者：{{ post.author_nickname || post.author_id || '匿名同学' }}</span>
          <span>{{ post.publish_time || '未知时间' }}</span>
        </div>
        <div class="detail-stats">
          <span>◎ {{ stats.view_count || 0 }}</span>
          <span>♥ {{ stats.like_count || 0 }}</span>
          <span class="mini-comment-meta"><i class="mini-comment-icon" aria-hidden="true"></i>{{ stats.comment_count || 0 }}</span>
          <span>★ {{ stats.collect_count || 0 }}</span>
        </div>
      </article>

      <div class="detail-actions">
        <button class="detail-action like-btn" :class="{ active: liked }" @click="sendAction('like')">
          <span>♥</span>
          <small>{{ stats.like_count || 0 }}</small>
        </button>
        <button class="detail-action collect-btn" :class="{ active: collected }" @click="sendAction('collect')">
          <span>★</span>
          <small>{{ stats.collect_count || 0 }}</small>
        </button>
        <button class="detail-action return-btn" @click="$router.push('/feed')">
          <span>↩</span>
          <small>返回</small>
        </button>
      </div>
      <p v-if="message" class="toast detail-toast">{{ message }}</p>

      <section class="comment-section">
        <h3>全部评论（{{ comments.length }}）</h3>
        <article v-for="comment in comments" :key="comment.comment_id" class="comment-card">
          <div class="comment-avatar">●</div>
          <div>
            <b>{{ comment.nickname || comment.user_id || '匿名同学' }}</b>
            <p>{{ comment.content || '这位同学什么也没说。' }}</p>
            <small>{{ comment.publish_time || '' }}</small>
          </div>
        </article>
        <EmptyState v-if="comments.length === 0" title="暂无评论" description="成为第一个交流的同学。" />
      </section>
    </template>

    <div v-if="selectedImage" class="image-lightbox" @click="closeImage">
      <button class="lightbox-close" type="button" @click.stop="closeImage">×</button>
      <img :src="selectedImage" alt="" @click.stop />
      <div class="lightbox-actions" @click.stop>
        <a :href="selectedImage" :download="selectedImageName">保存</a>
        <a :href="selectedImage" target="_blank" rel="noopener noreferrer">打开原图</a>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import { useRoute } from 'vue-router'
import EmptyState from '../components/EmptyState.vue'
import LoadingState from '../components/LoadingState.vue'
import TagBadge from '../components/TagBadge.vue'
import { DEFAULT_USER_ID, getPostDetail, recordBehavior } from '../api/request'
import { getPostStats, updatePostStats } from '../state/postStats'
import { imageUrlsOf } from '../utils/images'

const route = useRoute()
const userId = localStorage.getItem('campus_user_id') || DEFAULT_USER_ID
const post = ref({})
const stats = ref({})
const comments = ref([])
const loading = ref(false)
const error = ref('')
const message = ref('')
const liked = ref(false)
const collected = ref(false)
const selectedImage = ref('')

const tagList = computed(() =>
  String(post.value.tags || '')
    .split('|')
    .map((tag) => tag.trim())
    .filter(Boolean)
    .slice(0, 8)
)

const imageUrls = computed(() => imageUrlsOf(post.value))
const selectedImageName = computed(() => {
  if (!selectedImage.value) return 'post-image'
  return selectedImage.value.split('/').pop()?.split('@')[0] || 'post-image'
})

loadDetail()

async function loadDetail() {
  loading.value = true
  error.value = ''
  try {
    const detail = await getPostDetail(route.params.id)
    const cachedStats = getPostStats(route.params.id)
    post.value = detail.post || {}
    stats.value = {
      ...(detail.stats || {}),
      ...(cachedStats || {})
    }
    comments.value = detail.comments || []
    liked.value = Boolean(cachedStats?.liked)
    collected.value = Boolean(cachedStats?.collected)
    updatePostStats(route.params.id, {
      like_count: Number(stats.value.like_count || 0),
      collect_count: Number(stats.value.collect_count || 0),
      liked: liked.value,
      collected: collected.value
    })
    await recordBehavior({
      user_id: userId,
      post_id: String(route.params.id),
      action_type: 'view',
      dwell_time: 20,
      context: { time_period: '晚上' }
    })
  } catch (err) {
    error.value = err.message
  } finally {
    loading.value = false
  }
}

async function sendAction(actionType) {
  const previousLiked = liked.value
  const previousCollected = collected.value
  const previousLikeCount = Number(stats.value.like_count || 0)
  const previousCollectCount = Number(stats.value.collect_count || 0)
  let requestAction = actionType
  if (actionType === 'like') {
    liked.value = !previousLiked
    stats.value.like_count = Math.max(0, previousLikeCount + (liked.value ? 1 : -1))
    requestAction = liked.value ? 'like' : 'unlike'
  }
  if (actionType === 'collect') {
    collected.value = !previousCollected
    stats.value.collect_count = Math.max(0, previousCollectCount + (collected.value ? 1 : -1))
    requestAction = collected.value ? 'collect' : 'uncollect'
  }
  updatePostStats(route.params.id, {
    like_count: Number(stats.value.like_count || 0),
    collect_count: Number(stats.value.collect_count || 0),
    liked: liked.value,
    collected: collected.value
  })
  try {
    await recordBehavior({
      user_id: userId,
      post_id: String(route.params.id),
      action_type: requestAction,
      dwell_time: 30,
      context: { time_period: '晚上' }
    })
    message.value = actionType === 'like' ? (liked.value ? '已点赞' : '已取消点赞') : (collected.value ? '已收藏' : '已取消收藏')
    setTimeout(() => (message.value = ''), 1500)
  } catch (err) {
    if (actionType === 'like') {
      liked.value = previousLiked
      stats.value.like_count = previousLikeCount
    }
    if (actionType === 'collect') {
      collected.value = previousCollected
      stats.value.collect_count = previousCollectCount
    }
    updatePostStats(route.params.id, {
      like_count: previousLikeCount,
      collect_count: previousCollectCount,
      liked: previousLiked,
      collected: previousCollected
    })
    message.value = err.message
  }
}

function hideBrokenImage(event) {
  event.currentTarget.style.display = 'none'
}

function openImage(url) {
  selectedImage.value = url
}

function closeImage() {
  selectedImage.value = ''
}
</script>
