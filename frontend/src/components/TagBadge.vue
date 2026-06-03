<template>
  <span class="tag-badge">#{{ label || '校园' }}</span>
</template>

<script setup>
defineProps({
  label: {
    type: String,
    default: ''
  }
})
</script>
