<template>
  <div class="page feed-page" @wheel.prevent="handleWheel" @touchstart="touchStart" @touchend="touchEnd">
    <header class="top-header">
      <div>
        <h1>校园推荐</h1>
        <p>为你推荐 · 智能排序信息流</p>
      </div>
      <button class="ghost-btn" @click="loadRecommend">刷新</button>
    </header>

    <section class="user-switch">
      <input v-model="userInput" placeholder="输入 user_id 切换演示用户" />
      <button @click="applyUser">切换</button>
    </section>

    <LoadingState v-if="loading" />
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
        <span>{{ currentIndex + 1 }} / {{ posts.length }}</span>
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
import { DEFAULT_USER_ID, getRecommend, recordBehavior } from '../api/request'

const router = useRouter()
const userId = ref(localStorage.getItem('campus_user_id') || DEFAULT_USER_ID)
const userInput = ref(userId.value)
const posts = ref([])
const currentIndex = ref(0)
const loading = ref(false)
const error = ref('')
const toast = ref('')
const startY = ref(0)
let wheelLocked = false

const currentPost = computed(() => posts.value[currentIndex.value] || {})

loadRecommend()

async function loadRecommend() {
  loading.value = true
  error.value = ''
  toast.value = ''
  try {
    posts.value = await getRecommend(userId.value, 10)
    currentIndex.value = 0
  } catch (err) {
    error.value = err.message
  } finally {
    loading.value = false
  }
}

function applyUser() {
  const next = userInput.value.trim()
  if (!next) return
  userId.value = next
  localStorage.setItem('campus_user_id', next)
  loadRecommend()
}

function nextPost() {
  if (!posts.value.length) return
  currentIndex.value = Math.min(currentIndex.value + 1, posts.value.length - 1)
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
</script>
