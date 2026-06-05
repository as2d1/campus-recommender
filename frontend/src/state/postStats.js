import { reactive } from 'vue'

const postStats = reactive({})

function postIdOf(post) {
  return String(post?.post_id || '')
}

export function applyPostStats(post) {
  const postId = postIdOf(post)
  if (!postId) return post
  if (postStats[postId]) {
    Object.assign(post, postStats[postId])
  } else {
    postStats[postId] = {
      like_count: Number(post.like_count || 0),
      collect_count: Number(post.collect_count || 0),
      liked: Boolean(post.liked),
      collected: Boolean(post.collected)
    }
  }
  return post
}

export function getPostStats(postId) {
  return postStats[String(postId)] || null
}

export function updatePostStats(postId, patch) {
  const key = String(postId || '')
  if (!key) return
  postStats[key] = {
    ...(postStats[key] || {}),
    ...patch
  }
}
