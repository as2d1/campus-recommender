import axios from 'axios'

export const API_BASE_URL = 'http://127.0.0.1:8000'
export const DEFAULT_USER_ID =
  localStorage.getItem('campus_user_id') ||
  'dzQ5OWFZU1Rsb0NBdU5DMGlxR3dpSVMzYXJHR2VZZHBsNHVoYllWK3RjNnprSDJ4aWMzSWE1S21ycmlHaTd1bWZKS0FqNFoyb29lSWlXSjI='

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000
})

function unwrap(response) {
  const body = response.data
  if (!body || body.success !== true) {
    const message = body?.message || '请求失败'
    throw new Error(message)
  }
  return body.data
}

function friendlyError(error) {
  if (error.code === 'ERR_NETWORK' || error.message?.includes('Network')) {
    throw new Error('后端服务未连接，请确认 FastAPI 已在 http://127.0.0.1:8000 启动。')
  }
  throw error
}

export async function getRecommend(userId, topN = 10) {
  try {
    return unwrap(await client.get(`/api/recommend/${encodeURIComponent(userId)}`, { params: { top_n: topN } }))
  } catch (error) {
    friendlyError(error)
  }
}

export async function getHotPosts(topN = 20) {
  try {
    return unwrap(await client.get('/api/posts/hot', { params: { top_n: topN } }))
  } catch (error) {
    friendlyError(error)
  }
}

export async function getPostList(params = {}) {
  try {
    return unwrap(await client.get('/api/posts', { params }))
  } catch (error) {
    friendlyError(error)
  }
}

export async function getPostDetail(postId) {
  try {
    return unwrap(await client.get(`/api/posts/${encodeURIComponent(postId)}`))
  } catch (error) {
    friendlyError(error)
  }
}

export async function getUserProfile(userId) {
  try {
    return unwrap(await client.get(`/api/users/${encodeURIComponent(userId)}/profile`))
  } catch (error) {
    friendlyError(error)
  }
}

export async function recordBehavior(data) {
  try {
    return unwrap(await client.post('/api/behavior', data))
  } catch (error) {
    friendlyError(error)
  }
}

export async function refreshUserProfile(userId) {
  try {
    return unwrap(await client.post(`/api/users/${encodeURIComponent(userId)}/refresh-profile`))
  } catch (error) {
    friendlyError(error)
  }
}

export async function getBoards() {
  try {
    return unwrap(await client.get('/api/boards'))
  } catch (error) {
    friendlyError(error)
  }
}

export async function getTags() {
  try {
    return unwrap(await client.get('/api/tags'))
  } catch (error) {
    friendlyError(error)
  }
}

export async function createPost(data) {
  try {
    return unwrap(await client.post('/api/posts', data))
  } catch (error) {
    friendlyError(error)
  }
}
