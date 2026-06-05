<template>
  <nav class="bottom-nav">
    <RouterLink class="nav-item" to="/feed">
      <span>首页</span>
    </RouterLink>
    <RouterLink class="nav-item" to="/hot">
      <span>热榜</span>
    </RouterLink>
    <RouterLink class="publish-nav" to="/publish" aria-label="发布">
      <span>+</span>
    </RouterLink>
    <RouterLink class="nav-item" to="/message">
      <span>消息</span>
    </RouterLink>
    <RouterLink class="nav-item" to="/profile">
      <span>我的</span>
    </RouterLink>
  </nav>
</template>
