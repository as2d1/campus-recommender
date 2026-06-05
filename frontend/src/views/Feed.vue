<template>
  <div class="page feed-page" @wheel.prevent="handleWheel" @touchstart="touchStart" @touchend="touchEnd">
    <header class="top-header">
      <div>
        <h1>校园推荐</h1>
        <p>为你推荐 · 智能排序信息流</p>
      </div>
      <button class="ghost-btn" @click="checkUserAndLoad">刷新</button>
    </header>

    <section class="user-switch">
      <input v-model="userInput" placeholder="输入 user_id 切换演示用户" />
      <button @click="applyUser">切换</button>
    </section>

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

    <section v-else class="feed-stage">
      <FeedCard :post="currentPost" @open="openDetail" />
      <ActionBar
        :post="currentPost"
        @like="sendAction('like')"
        @collect="sendAction('collect')"
        @comment="openDetail(currentPost.post_id)"
        @skip="skipPost"
        @detail="openDetail(currentPost.post_id)"
      />
      <div class="pager-controls">
        <button @click="prevPost">上一条</button>
        <span>{{ currentIndex + 1 }} / {{ posts.length }}{{ loadingMore ? ' · 加载中' : '' }}</span>
        <button @click="nextPost">下一条</button>
      </div>
      <p v-if="toast" class="toast">{{ toast }}</p>
    </section>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import ActionBar from '../components/ActionBar.vue'
import EmptyState from '../components/EmptyState.vue'
import FeedCard from '../components/FeedCard.vue'
import LoadingState from '../components/LoadingState.vue'
import { DEFAULT_USER_ID, getRecommend, getTaxonomy, getUserStatus, recordBehavior, saveUserPreferences } from '../api/request'

const router = useRouter()
const userId = ref(localStorage.getItem('campus_user_id') || DEFAULT_USER_ID)
const userInput = ref(userId.value)
const posts = ref([])
const currentIndex = ref(0)
const loading = ref(false)
const loadingMore = ref(false)
const error = ref('')
const toast = ref('')
const startY = ref(0)
const onboarding = ref(false)
const selectedBoards = ref([])
const selectedTags = ref([])
let wheelLocked = false

const boards = ref([])
const seedTags = ref([])

const currentPost = computed(() => posts.value[currentIndex.value] || {})

checkUserAndLoad()

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
      currentIndex.value = 0
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
    const nextPosts = await getRecommend(userId.value, 10)
    if (append) {
      const seen = new Set(posts.value.map((post) => String(post.post_id)))
      posts.value.push(...nextPosts.filter((post) => !seen.has(String(post.post_id))))
    } else {
      posts.value = nextPosts
      currentIndex.value = 0
    }
  } catch (err) {
    error.value = err.message
  } finally {
    loading.value = false
    loadingMore.value = false
  }
}

function applyUser() {
  const next = userInput.value.trim()
  if (!next) return
  userId.value = next
  localStorage.setItem('campus_user_id', next)
  selectedBoards.value = []
  selectedTags.value = []
  checkUserAndLoad()
}

async function nextPost() {
  if (!posts.value.length) return
  if (currentIndex.value >= posts.value.length - 1) {
    if (!loadingMore.value) {
      await loadRecommend({ append: true })
    }
    if (posts.value.length > currentIndex.value + 1) {
      currentIndex.value += 1
    }
    return
  }
  currentIndex.value = Math.min(currentIndex.value + 1, posts.value.length - 1)
  if (posts.value.length - currentIndex.value <= 3 && !loadingMore.value) {
    loadRecommend({ append: true })
  }
}

function prevPost() {
  if (!posts.value.length) return
  currentIndex.value = Math.max(currentIndex.value - 1, 0)
}

function handleWheel(event) {
  if (wheelLocked) return
  wheelLocked = true
  event.deltaY > 0 ? nextPost() : prevPost()
  setTimeout(() => {
    wheelLocked = false
  }, 420)
}

function touchStart(event) {
  startY.value = event.changedTouches[0].clientY
}

function touchEnd(event) {
  const delta = startY.value - event.changedTouches[0].clientY
  if (Math.abs(delta) < 40) return
  delta > 0 ? nextPost() : prevPost()
}

async function sendAction(actionType) {
  if (!currentPost.value.post_id) return
  try {
    await recordBehavior({
      user_id: userId.value,
      post_id: String(currentPost.value.post_id),
      action_type: actionType,
      dwell_time: actionType === 'skip' ? 2 : 20,
      context: {
        time_period: '晚上'
      }
    })
    toast.value = actionType === 'like' ? '已点赞，行为已写入画像' : actionType === 'collect' ? '已收藏' : '已跳过'
    if (actionType === 'like') currentPost.value.like_count = Number(currentPost.value.like_count || 0) + 1
    if (actionType === 'collect') currentPost.value.collect_count = Number(currentPost.value.collect_count || 0) + 1
    if (actionType === 'skip') nextPost()
    setTimeout(() => (toast.value = ''), 1800)
  } catch (err) {
    toast.value = err.message
  }
}

function skipPost() {
  sendAction('skip')
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
