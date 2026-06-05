const CDN_BASE = 'https://b1.cdn.zanao.com/'
const CDN_SUFFIX = '@!common'

export function imageUrlOf(value) {
  const text = String(value || '').trim()
  if (!text) return ''
  if (text.startsWith('http://') || text.startsWith('https://')) {
    if (text.includes('cdn.zanao.com/') && !text.split('/').pop().includes('@')) {
      return `${text}${CDN_SUFFIX}`
    }
    return text
  }
  return `${CDN_BASE}${text.replace(/^\/+/, '')}${CDN_SUFFIX}`
}

export function imageUrlsOf(post) {
  if (Array.isArray(post?.image_urls) && post.image_urls.length) {
    return post.image_urls.map((url) => imageUrlOf(url)).filter(Boolean)
  }
  if (Array.isArray(post?.image_paths) && post.image_paths.length) {
    return post.image_paths.map((path) => imageUrlOf(path)).filter(Boolean)
  }
  return []
}

export function parseImageInput(value) {
  return String(value || '')
    .split(/[\n,，]+/)
    .map((item) => item.trim())
    .filter(Boolean)
}
