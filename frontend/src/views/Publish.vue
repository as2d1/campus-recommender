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
      <label>
        <span>图片路径（选填）</span>
        <textarea
          v-model="form.imagePaths"
          class="image-path-input"
          maxlength="1200"
          placeholder="每行一张图片，如：upload/2026/06/03/pkuuttjeex42lhb.jpeg"
        ></textarea>
        <small>{{ imagePaths.length }} 张</small>
      </label>
      <div class="switch-row">
        <span>匿名发布</span>
        <button type="button" class="switch" :class="{ on: form.anonymous }" @click="form.anonymous = !form.anonymous">
          <i></i>
        </button>
      </div>
      <div class="upload-placeholder image-preview-row">
        <span>▣</span>
        <div>
          <p>添加图片</p>
          <small>支持 zanao 相对路径或完整图片链接</small>
        </div>
      </div>
      <div v-if="previewUrls.length" class="post-image-grid preview">
        <img v-for="url in previewUrls" :key="url" :src="url" alt="" loading="lazy" @error="hideBrokenImage" />
      </div>
      <p v-if="message" class="form-message">{{ message }}</p>
      <button class="primary-btn full-btn" type="submit">发布到校园论坛</button>
    </form>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { DEFAULT_USER_ID, createPost, getTaxonomy } from '../api/request'
import { imageUrlOf, parseImageInput } from '../utils/images'

const router = useRouter()
const message = ref('')
const boards = ref([])
const form = reactive({
  board: '',
  title: '',
  content: '',
  tags: '',
  imagePaths: '',
  anonymous: false
})

const imagePaths = computed(() => parseImageInput(form.imagePaths))
const previewUrls = computed(() => imagePaths.value.map((path) => imageUrlOf(path)).filter(Boolean).slice(0, 6))

onMounted(loadTaxonomy)

async function loadTaxonomy() {
  try {
    const taxonomy = await getTaxonomy()
    if (Array.isArray(taxonomy.boards) && taxonomy.boards.length) {
      boards.value = taxonomy.boards
      if (!boards.value.includes(form.board)) {
        form.board = boards.value[0]
      }
    }
  } catch {
    // Keep fallback boards when the backend is unavailable.
  }
}

async function submitPost() {
  if (!form.board) {
    message.value = '板块加载中，请稍后再发布。'
    return
  }
  if (form.title.trim().length < 2 || form.content.trim().length < 5) {
    message.value = '标题至少 2 个字，正文至少 5 个字。'
    return
  }
  const payload = {
    user_id: localStorage.getItem('campus_user_id') || DEFAULT_USER_ID,
    board: form.board,
    title: form.title,
    content: form.content,
    tags: form.tags,
    anonymous: form.anonymous,
    image_paths: imagePaths.value,
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

function hideBrokenImage(event) {
  event.currentTarget.style.display = 'none'
}
</script>
