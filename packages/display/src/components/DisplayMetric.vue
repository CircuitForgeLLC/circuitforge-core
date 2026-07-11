<script setup lang="ts">
import { computed } from 'vue'
import type { DisplaySeverity } from '../theme'

const props = withDefaults(
  defineProps<{
    value: string | number
    label: string
    unit?: string
    severity?: DisplaySeverity
    sparkline?: number[]
  }>(),
  {
    unit: '',
    severity: 'ok',
    sparkline: undefined,
  },
)

const displayValue = computed(() => `${props.value}${props.unit ? props.unit : ''}`)

const sparklinePoints = computed(() => {
  const data = props.sparkline
  if (!data || data.length < 2) return null
  const min = Math.min(...data)
  const max = Math.max(...data)
  const range = max - min || 1
  const width = 100
  const height = 24
  const step = width / (data.length - 1)
  return data
    .map((v, i) => {
      const x = i * step
      const y = height - ((v - min) / range) * height
      return `${x.toFixed(2)},${y.toFixed(2)}`
    })
    .join(' ')
})
</script>

<template>
  <div class="cf-display-metric" :class="`cf-display-metric--${severity}`" role="group" :aria-label="label">
    <div class="cf-display-metric__value">{{ displayValue }}</div>
    <div class="cf-display-metric__label">{{ label }}</div>
    <svg
      v-if="sparklinePoints"
      class="cf-display-metric__sparkline"
      viewBox="0 0 100 24"
      preserveAspectRatio="none"
      aria-hidden="true"
    >
      <polyline :points="sparklinePoints" fill="none" stroke-width="2" />
    </svg>
  </div>
</template>

<style scoped>
.cf-display-metric {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.15em;
  min-width: 6rem;
  padding: 0.5rem 0.75rem;
  border-radius: 0.5rem;
  background: var(--cf-display-surface-raised, #161b22);
  color: var(--cf-display-text-primary, #e6edf3);
}

.cf-display-metric__value {
  font-size: clamp(1.1rem, 4vw, 2rem);
  font-weight: 700;
  line-height: 1.1;
}

.cf-display-metric__label {
  font-size: clamp(0.6rem, 1.5vw, 0.85rem);
  color: var(--cf-display-text-muted, #8b949e);
  white-space: nowrap;
}

.cf-display-metric__sparkline {
  width: 100%;
  height: 1.2rem;
  margin-top: 0.2rem;
}

.cf-display-metric__sparkline polyline {
  stroke: currentColor;
}

.cf-display-metric--ok {
  color: var(--cf-display-sev-ok, #3fb950);
}

.cf-display-metric--warn {
  color: var(--cf-display-sev-warn, #d29922);
}

.cf-display-metric--crit {
  color: var(--cf-display-sev-crit, #f85149);
}

.cf-display-metric--ok .cf-display-metric__value,
.cf-display-metric--warn .cf-display-metric__value,
.cf-display-metric--crit .cf-display-metric__value {
  color: currentColor;
}
</style>
