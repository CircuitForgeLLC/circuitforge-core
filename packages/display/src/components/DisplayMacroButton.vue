<script setup lang="ts">
export type MacroAction =
  | { type: 'shell'; command: string }
  | { type: 'url'; url: string }
  | { type: 'api'; endpoint: string; method?: string; body?: unknown }
  | { type: 'display_switch'; target: string }

const props = defineProps<{
  icon: string
  label: string
  action: MacroAction
  disabled?: boolean
}>()

const emit = defineEmits<{
  trigger: [action: MacroAction]
}>()

function onActivate() {
  if (props.disabled) return
  emit('trigger', props.action)
}
</script>

<template>
  <button
    type="button"
    class="cf-display-macro"
    :disabled="disabled"
    :aria-label="label"
    @click="onActivate"
  >
    <span class="cf-display-macro__icon" aria-hidden="true">{{ icon }}</span>
    <span class="cf-display-macro__label">{{ label }}</span>
  </button>
</template>

<style scoped>
.cf-display-macro {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.15em;
  /* 44px is the accessibility-minimum touch target; strip displays prefer 64px+ */
  min-width: 4rem;
  min-height: 4rem;
  padding: 0.5rem;
  border: 1px solid var(--cf-display-surface-border, #30363d);
  border-radius: 0.75rem;
  background: var(--cf-display-surface-raised, #161b22);
  color: var(--cf-display-text-primary, #e6edf3);
  cursor: pointer;
  user-select: none;
  touch-action: manipulation;
}

.cf-display-macro:hover:not(:disabled),
.cf-display-macro:focus-visible {
  border-color: var(--cf-display-accent, #39d353);
  outline: none;
}

.cf-display-macro:active:not(:disabled) {
  background: var(--cf-display-surface, #0d1117);
}

.cf-display-macro:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.cf-display-macro__icon {
  font-size: clamp(1.2rem, 3vw, 1.8rem);
  line-height: 1;
}

.cf-display-macro__label {
  font-size: clamp(0.6rem, 1.4vw, 0.8rem);
  text-align: center;
  color: var(--cf-display-text-muted, #8b949e);
}
</style>
