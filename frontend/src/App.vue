<template>
  <div class="app-bg">
    <main class="phone-shell">
      <div class="status-bar">
        <span>9:41</span>
        <span class="status-icons">▮▮▮  WiFi  ▰</span>
      </div>
      <section class="phone-screen">
        <RouterView />
      </section>
      <BottomNav />
    </main>
  </div>
</template>

<script setup>
import BottomNav from './components/BottomNav.vue'
</script>
