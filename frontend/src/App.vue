<template>
  <div class="app-bg">
    <main class="phone-shell">
      <section class="phone-screen">
        <RouterView v-slot="{ Component }">
          <KeepAlive include="Feed">
            <component :is="Component" />
          </KeepAlive>
        </RouterView>
      </section>
      <BottomNav />
    </main>
  </div>
</template>

<script setup>
import BottomNav from './components/BottomNav.vue'
</script>
