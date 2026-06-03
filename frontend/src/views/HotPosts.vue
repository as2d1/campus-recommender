<template>
  <div class="page">
    <header class="top-header">
      <div>
        <h1>热门火文</h1>
        <p>校园里正在被围观的内容</p>
      </div>
      <button class="ghost-btn" @click="loadHot">刷新</button>
    </header>

    <div class="tabs">
      <button v-for="tab in tabs" :key="tab" :class="{ active: activeTab === tab }" @click="activeTab = tab">{{ tab }}</button>
    </div>

    <LoadingState v-if="loading" text="正在加载热门榜..." />
    <EmptyState v-else-if="error" title="热榜加载失败" :description="error" button-text="重试" @action="loadHot" />
    <EmptyState v-else-if="posts.length === 0" title="暂无热门帖子" description="等同学们互动起来，热榜就会亮起来。" />

    <section v-else class="rank-list">
      <article v-for="(post, index) in posts" :key="post.post_id" class="rank-item" @click="openDetail(post.post_id)">
        <div class="rank-no" :class="`top-${index + 1}`">{{ index + 1 }}</div>
        <div class="rank-body">
          <h3>{{ post.title || '无标题帖子' }}</h3>
          <div class="rank-meta">
            <span>{{ post.board || '校园论坛' }}</span>
            <span>◎ {{ post.view_count || 0 }}</span>
            <span>♥ {{ post.like_count || 0 }}</span>
            <span>□ {{ post.comment_count || 0 }}</span>
          </div>
        </div>
      </article>
    </section>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import EmptyState from '../components/EmptyState.vue'
import LoadingState from '../components/LoadingState.vue'
import { getHotPosts } from '../api/request'

const router = useRouter()
const tabs = ['24小时', '7天', '30天']
const activeTab = ref('24小时')
const posts = ref([])
const loading = ref(false)
const error = ref('')

loadHot()

async function loadHot() {
  loading.value = true
  error.value = ''
  try {
    posts.value = await getHotPosts(20)
  } catch (err) {
    error.value = err.message
  } finally {
    loading.value = false
  }
}

function openDetail(postId) {
  router.push(`/post/${encodeURIComponent(postId)}`)
}
</script>
