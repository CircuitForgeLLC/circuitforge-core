<script setup lang="ts">
import { computed } from 'vue'
import type { DisplaySeverity } from '../theme'

const props = withDefaults(
  defineProps<{
    message: string
    timestamp: string | Date
    severity?: DisplaySeverity
  }>(),
  {
    severity: 'ok',
  },
)

const formattedTime = computed(() => {
  const d = typeof props.timestamp === 'string' ? new Date(props.timestamp) : props.timestamp
  if (Number.isNaN(d.getTime())) return String(props.timestamp)
  return d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
})
</script>

<template>
  <div class="cf-display-alert" :class="`cf-display-alert--${severity}`" role="status">
    <span class="cf-display-alert__dot" aria-hidden="true" />
    <span class="cf-display-alert__time">{{ formattedTime }}</span>
    <span class="cf-display-alert__message">{{ message }}</span>
  </div>
</template>

<style scoped>
.cf-display-alert {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.35rem 0.5rem;
  border-left: 4px solid currentColor;
  background: var(--cf-display-surface-raised, #161b22);
  color: var(--cf-display-text-primary, #e6edf3);
  font-size: clamp(0.65rem, 1.6vw, 0.9rem);
  overflow: hidden;
}

.cf-display-alert__dot {
  width: 0.5em;
  height: 0.5em;
  border-radius: 50%;
  background: currentColor;
  flex-shrink: 0;
}

.cf-display-alert__time {
  color: var(--cf-display-text-muted, #8b949e);
  flex-shrink: 0;
  font-variant-numeric: tabular-nums;
}

.cf-display-alert__message {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.cf-display-alert--ok {
  color: var(--cf-display-text-muted, #8b949e);
}

.cf-display-alert--warn {
  color: var(--cf-display-sev-warn, #d29922);
}

.cf-display-alert--crit {
  color: var(--cf-display-sev-crit, #f85149);
}
</style>
