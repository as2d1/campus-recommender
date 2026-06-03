<template>
  <div class="page">
    <header class="top-header">
      <div>
        <h1>消息</h1>
        <p>评论、点赞与系统通知</p>
      </div>
    </header>
    <div class="tabs">
      <button v-for="tab in tabs" :key="tab" :class="{ active: activeTab === tab }" @click="activeTab = tab">{{ tab }}</button>
    </div>
    <section class="message-list">
      <article v-for="item in filteredMessages" :key="item.id" class="message-card">
        <span class="message-icon">{{ item.icon }}</span>
        <div>
          <h3>{{ item.title }}</h3>
          <p>{{ item.content }}</p>
        </div>
        <small>{{ item.time }}</small>
      </article>
    </section>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'

const tabs = ['全部', '评论', '点赞', '系统']
const activeTab = ref('全部')
const messages = [
  { id: 1, type: '评论', icon: '◌', title: '评论通知', content: 'u_0456 评论了你的帖子：期末重点有吗？', time: '2分钟前' },
  { id: 2, type: '点赞', icon: '★', title: '点赞通知', content: 'u_0789 赞了你的帖子：数据结构复习资料整理', time: '10分钟前' },
  { id: 3, type: '系统', icon: '✦', title: '系统通知', content: '你的用户画像已更新，点击查看推荐变化。', time: '1小时前' },
  { id: 4, type: '系统', icon: '⚑', title: '推荐更新通知', content: '有新的优质内容进入你的推荐流。', time: '2小时前' }
]

const filteredMessages = computed(() =>
  activeTab.value === '全部' ? messages : messages.filter((item) => item.type === activeTab.value)
)
</script>
