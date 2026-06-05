<template>
  <div class="page profile-page">
    <header class="profile-hero">
      <button class="ghost-icon">◌</button>
      <div class="avatar">{{ avatarInitial }}</div>
      <h1>{{ displayName }}</h1>
      <section class="user-switch profile-switch">
        <input v-model="userInput" placeholder="输入 user_id" />
        <button @click="applyUser">切换</button>
      </section>
    </header>

    <LoadingState v-if="loading" text="正在加载用户画像..." />
    <EmptyState v-else-if="error" title="画像加载失败" :description="error" button-text="重试" @action="loadProfile" />

    <template v-else>
      <section class="panel">
        <h2>我的兴趣画像</h2>
        <div class="profile-tags">
          <TagBadge v-for="tag in tags(profile.long_term_tags)" :key="`l-${tag}`" :label="tag" />
          <TagBadge v-for="tag in tags(profile.short_term_tags)" :key="`s-${tag}`" :label="tag" />
        </div>
        <p class="muted">常看板块：{{ profile.preferred_boards || '暂无' }}</p>
      </section>

      <section class="panel">
        <h2>兴趣权重</h2>
        <InterestBar label="学习兴趣" :value="profile.learning_interest_weight" />
        <InterestBar label="生活兴趣" :value="profile.life_interest_weight" color="linear-gradient(90deg, #d39a65, #e4bf86)" />
        <InterestBar label="社交兴趣" :value="profile.social_interest_weight" color="linear-gradient(90deg, #8baac7, #afa2ca)" />
        <InterestBar label="发展兴趣" :value="profile.career_interest_weight" color="linear-gradient(90deg, #7aa08a, #91b9a5)" />
      </section>

      <section class="menu-list">
        <button>我的发布 <span>›</span></button>
        <button>我的收藏 <span>›</span></button>
        <button>我的点赞 <span>›</span></button>
        <button>浏览历史 <span>›</span></button>
        <button @click="refreshProfile">重新画像 <span>↻</span></button>
      </section>
      <p v-if="message" class="toast profile-toast">{{ message }}</p>
    </template>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import EmptyState from '../components/EmptyState.vue'
import InterestBar from '../components/InterestBar.vue'
import LoadingState from '../components/LoadingState.vue'
import TagBadge from '../components/TagBadge.vue'
import { DEFAULT_USER_ID, getUserProfile, refreshUserProfile } from '../api/request'

const userId = ref(localStorage.getItem('campus_user_id') || DEFAULT_USER_ID)
const userInput = ref(userId.value)
const profile = ref({})
const loading = ref(false)
const error = ref('')
const message = ref('')

const shortUserId = computed(() => {
  const id = String(profile.value.user_id || userId.value || 'u_0001')
  return id.length > 14 ? `${id.slice(0, 8)}...${id.slice(-4)}` : id
})

const displayName = computed(() => profile.value.nickname || shortUserId.value)

const avatarInitial = computed(() => {
  const name = String(profile.value.nickname || profile.value.user_id || userId.value || '?').trim()
  return (name.match(/[\u4e00-\u9fa5A-Za-z0-9]/)?.[0] || '?').toUpperCase()
})

loadProfile()

async function loadProfile() {
  loading.value = true
  error.value = ''
  try {
    profile.value = await getUserProfile(userId.value)
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
  loadProfile()
}

async function refreshProfile() {
  try {
    profile.value = await refreshUserProfile(userId.value)
    message.value = '画像已重新计算'
    setTimeout(() => (message.value = ''), 1600)
  } catch (err) {
    message.value = err.message
  }
}

function tags(value) {
  return String(value || '')
    .split('|')
    .map((tag) => tag.trim())
    .filter(Boolean)
    .slice(0, 12)
}
</script>
