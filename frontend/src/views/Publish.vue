<template>
  <div class="page form-page">
    <header class="top-header">
      <button class="back-btn" @click="$router.back()">‹</button>
      <div>
        <h1>发布帖子</h1>
        <p>分享你的校园想法</p>
      </div>
      <button class="publish-mini" @click="submitPost">发布</button>
    </header>

    <form class="publish-form" @submit.prevent="submitPost">
      <label>
        <span>选择板块</span>
        <select v-model="form.board">
          <option v-for="board in boards" :key="board" :value="board">{{ board }}</option>
        </select>
      </label>
      <label>
        <span>标题</span>
        <input v-model="form.title" maxlength="30" placeholder="请输入标题（2-30字）" />
        <small>{{ form.title.length }}/30</small>
      </label>
      <label>
        <span>正文</span>
        <textarea v-model="form.content" maxlength="1000" placeholder="分享你的想法..."></textarea>
        <small>{{ form.content.length }}/1000</small>
      </label>
      <label>
        <span>标签（选填）</span>
        <input v-model="form.tags" placeholder="点击输入标签，回车添加，如：期末|课程评价" />
      </label>
      <div class="switch-row">
        <span>匿名发布</span>
        <button type="button" class="switch" :class="{ on: form.anonymous }" @click="form.anonymous = !form.anonymous">
          <i></i>
        </button>
      </div>
      <div class="upload-placeholder">
        <span>▣</span>
        <p>添加图片（占位）</p>
      </div>
      <p v-if="message" class="form-message">{{ message }}</p>
      <button class="primary-btn full-btn" type="submit">发布到校园论坛</button>
    </form>
  </div>
</template>

<script setup>
import { reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { createPost } from '../api/request'

const router = useRouter()
const message = ref('')
const boards = ['打听求助', '恋爱交友', '校园趣事', '兼职招聘', '校园招聘', '二手闲置']
const form = reactive({
  board: boards[0],
  title: '',
  content: '',
  tags: '',
  anonymous: false
})

async function submitPost() {
  if (form.title.trim().length < 2 || form.content.trim().length < 5) {
    message.value = '标题至少 2 个字，正文至少 5 个字。'
    return
  }
  const payload = {
    ...form,
    created_at: new Date().toISOString()
  }
  try {
    await createPost(payload)
    message.value = '发布成功。'
  } catch {
    const localPosts = JSON.parse(localStorage.getItem('campus_local_posts') || '[]')
    localPosts.unshift({ ...payload, post_id: `local_${Date.now()}` })
    localStorage.setItem('campus_local_posts', JSON.stringify(localPosts))
    message.value = '当前为前端模拟发布，后续可接入后端发帖接口。'
  }
  setTimeout(() => router.push('/feed'), 900)
}
</script>
