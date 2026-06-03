import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/', redirect: '/feed' },
  { path: '/feed', name: 'feed', component: () => import('../views/Feed.vue') },
  { path: '/hot', name: 'hot', component: () => import('../views/HotPosts.vue') },
  { path: '/publish', name: 'publish', component: () => import('../views/Publish.vue') },
  { path: '/message', name: 'message', component: () => import('../views/Message.vue') },
  { path: '/profile', name: 'profile', component: () => import('../views/Profile.vue') },
  { path: '/post/:id', name: 'post-detail', component: () => import('../views/PostDetail.vue') }
]

export default createRouter({
  history: createWebHistory(),
  routes
})
