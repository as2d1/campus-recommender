<template>
  <div class="page feed-page">
    <header class="top-header">
      <div>
        <h1>校园推荐</h1>
        <p>为你推荐 · 智能排序信息流</p>
      </div>
    </header>

    <section v-if="onboarding" class="panel onboarding-panel">
      <h2>选择你的兴趣</h2>
      <p class="muted">选几个感兴趣的板块和话题，先给你一批更贴近的内容。</p>
      <div class="choice-grid">
        <button
          v-for="board in boards"
          :key="board"
          type="button"
          class="choice-chip"
          :class="{ active: selectedBoards.includes(board) }"
          @click="toggleChoice(selectedBoards, board)"
        >
          {{ board }}
        </button>
      </div>
      <div class="choice-grid compact">
        <button
          v-for="tag in seedTags"
          :key="tag"
          type="button"
          class="choice-chip"
          :class="{ active: selectedTags.includes(tag) }"
          @click="toggleChoice(selectedTags, tag)"
        >
          {{ tag }}
        </button>
      </div>
      <p v-if="toast" class="inline-toast">{{ toast }}</p>
      <button type="button" class="primary-btn full-btn" :disabled="loading" @click="submitPreferences">
        {{ loading ? '生成中...' : '开始推荐' }}
      </button>
    </section>

    <LoadingState v-if="loading && !onboarding" />
    <EmptyState
      v-else-if="error"
      title="推荐加载失败"
      :description="error"
      button-text="重试"
      @action="loadRecommend"
    />
    <EmptyState
      v-else-if="posts.length === 0"
      title="暂无推荐内容"
      description="暂无推荐内容，点击换一批试试。"
      button-text="换一批"
      @action="loadRecommend"
    />

    <section v-else class="feed-list">
      <article v-for="post in posts" :key="post.post_id" class="feed-item">
        <FeedCard :post="post" @open="openDetail" />
        <ActionBar
          :post="post"
          @like="sendAction('like', post)"
          @collect="sendAction('collect', post)"
          @comment="openDetail(post.post_id)"
          @skip="skipPost(post)"
          @detail="openDetail(post.post_id)"
        />
      </article>
      <div class="pager-controls">
        <button :disabled="loadingMore" @click="loadRecommend({ append: true })">
          {{ loadingMore ? '加载中...' : '加载更多' }}
        </button>
      </div>
      <p v-if="toast" class="toast">{{ toast }}</p>
    </section>
  </div>
</template>

<script setup>
import { onActivated, onMounted, onUnmounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import ActionBar from '../components/ActionBar.vue'
import EmptyState from '../components/EmptyState.vue'
import FeedCard from '../components/FeedCard.vue'
import LoadingState from '../components/LoadingState.vue'
import { DEFAULT_USER_ID, getRecommend, getTaxonomy, getUserStatus, recordBehavior, saveUserPreferences } from '../api/request'
import { applyPostStats, updatePostStats } from '../state/postStats'

defineOptions({ name: 'Feed' })

const router = useRouter()
const userId = ref(localStorage.getItem('campus_user_id') || DEFAULT_USER_ID)
const posts = ref([])
const loading = ref(false)
const loadingMore = ref(false)
const error = ref('')
const toast = ref('')
const onboarding = ref(false)
const selectedBoards = ref([])
const selectedTags = ref([])

const boards = ref([])
const seedTags = ref([])

checkUserAndLoad()

onActivated(() => {
  syncUserAndReload()
})

onMounted(() => {
  window.addEventListener('campus-user-change', syncUserAndReload)
})

onUnmounted(() => {
  window.removeEventListener('campus-user-change', syncUserAndReload)
})

function syncUserAndReload() {
  const latestUserId = localStorage.getItem('campus_user_id') || DEFAULT_USER_ID
  if (latestUserId !== userId.value) {
    userId.value = latestUserId
    posts.value = []
    selectedBoards.value = []
    selectedTags.value = []
    checkUserAndLoad()
    return
  }
  posts.value.forEach((post) => applyPostStats(post))
}

async function checkUserAndLoad() {
  loading.value = true
  error.value = ''
  toast.value = ''
  try {
    await loadTaxonomy()
    const status = await getUserStatus(userId.value)
    onboarding.value = Boolean(status.needs_onboarding)
    if (onboarding.value) {
      posts.value = []
    } else {
      await loadRecommend({ append: false })
    }
  } catch (err) {
    error.value = err.message
  } finally {
    loading.value = false
  }
}

async function loadTaxonomy() {
  const taxonomy = await getTaxonomy()
  if (Array.isArray(taxonomy.boards) && taxonomy.boards.length) {
    boards.value = taxonomy.boards
  }
  if (Array.isArray(taxonomy.cold_start_tags) && taxonomy.cold_start_tags.length) {
    seedTags.value = taxonomy.cold_start_tags
  }
}

async function loadRecommend({ append = false } = {}) {
  if (append) loadingMore.value = true
  else loading.value = true
  error.value = ''
  toast.value = ''
  try {
    const nextPosts = (await getRecommend(userId.value, 10)).map((post) => applyPostStats(post))
    if (append) {
      const seen = new Set(posts.value.map((post) => String(post.post_id)))
      posts.value.push(...nextPosts.filter((post) => !seen.has(String(post.post_id))))
    } else {
      posts.value = nextPosts
    }
  } catch (err) {
    error.value = err.message
  } finally {
    loading.value = false
    loadingMore.value = false
  }
}

async function sendAction(actionType, post) {
  if (!post?.post_id) return
  const previousLiked = Boolean(post.liked)
  const previousCollected = Boolean(post.collected)
  const previousLikeCount = Number(post.like_count || 0)
  const previousCollectCount = Number(post.collect_count || 0)
  const skippedIndex = actionType === 'skip'
    ? posts.value.findIndex((item) => String(item.post_id) === String(post.post_id))
    : -1
  let requestAction = actionType
  if (actionType === 'like') {
    post.liked = !previousLiked
    post.like_count = Math.max(0, previousLikeCount + (post.liked ? 1 : -1))
    requestAction = post.liked ? 'like' : 'unlike'
  }
  if (actionType === 'collect') {
    post.collected = !previousCollected
    post.collect_count = Math.max(0, previousCollectCount + (post.collected ? 1 : -1))
    requestAction = post.collected ? 'collect' : 'uncollect'
  }
  if (actionType === 'skip' && skippedIndex >= 0) {
    posts.value.splice(skippedIndex, 1)
  }
  updatePostStats(post.post_id, {
    like_count: Number(post.like_count || 0),
    collect_count: Number(post.collect_count || 0),
    liked: Boolean(post.liked),
    collected: Boolean(post.collected)
  })
  try {
    await recordBehavior({
      user_id: userId.value,
      post_id: String(post.post_id),
      action_type: requestAction,
      dwell_time: actionType === 'skip' ? 2 : 20,
      context: {
        time_period: '晚上'
      }
    })
    toast.value = actionType === 'like' ? (post.liked ? '已点赞' : '已取消点赞') : actionType === 'collect' ? (post.collected ? '已收藏' : '已取消收藏') : '已跳过'
    setTimeout(() => (toast.value = ''), 1800)
  } catch (err) {
    if (actionType === 'like') {
      post.liked = previousLiked
      post.like_count = previousLikeCount
    }
    if (actionType === 'collect') {
      post.collected = previousCollected
      post.collect_count = previousCollectCount
    }
    if (actionType === 'skip' && skippedIndex >= 0) {
      posts.value.splice(skippedIndex, 0, post)
    }
    updatePostStats(post.post_id, {
      like_count: previousLikeCount,
      collect_count: previousCollectCount,
      liked: previousLiked,
      collected: previousCollected
    })
    toast.value = err.message
  }
}

function skipPost(post) {
  sendAction('skip', post)
}

function openDetail(postId) {
  if (postId) router.push(`/post/${encodeURIComponent(postId)}`)
}

function toggleChoice(list, value) {
  const index = list.indexOf(value)
  if (index >= 0) list.splice(index, 1)
  else list.push(value)
}

async function submitPreferences() {
  if (!selectedBoards.value.length && !selectedTags.value.length) {
    toast.value = '请至少选择一个板块或标签'
    return
  }
  loading.value = true
  try {
    await saveUserPreferences(userId.value, {
      selected_boards: selectedBoards.value,
      selected_tags: selectedTags.value
    })
    onboarding.value = false
    await loadRecommend({ append: false })
  } catch (err) {
    error.value = err.message
  } finally {
    loading.value = false
  }
}
</script>
