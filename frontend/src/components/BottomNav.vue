<template>
  <nav class="bottom-nav">
    <RouterLink class="nav-item" to="/feed">
      <span class="nav-icon">⌂</span>
      <span>首页</span>
    </RouterLink>
    <RouterLink class="nav-item" to="/hot">
      <span class="nav-icon">♨</span>
      <span>热榜</span>
    </RouterLink>
    <RouterLink class="publish-nav" to="/publish" aria-label="发布">
      <span>+</span>
    </RouterLink>
    <RouterLink class="nav-item" to="/message">
      <span class="nav-icon">▣</span>
      <span>消息</span>
    </RouterLink>
    <RouterLink class="nav-item" to="/profile">
      <span class="nav-icon">♙</span>
      <span>我的</span>
    </RouterLink>
  </nav>
</template>
