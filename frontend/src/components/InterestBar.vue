<template>
  <div class="interest-bar">
    <div class="interest-label">
      <span>{{ label }}</span>
      <b>{{ scoreText }}</b>
    </div>
    <div class="bar-track">
      <span class="bar-fill" :style="{ width: width, background: color }"></span>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  label: { type: String, required: true },
  value: { type: [Number, String], default: 0 },
  color: { type: String, default: 'linear-gradient(90deg, #7aa08a, #8baac7)' }
})

const normalized = computed(() => Math.max(0, Math.min(1, Number(props.value || 0))))
const width = computed(() => `${Math.max(4, normalized.value * 100)}%`)
const scoreText = computed(() => normalized.value.toFixed(2))
</script>
