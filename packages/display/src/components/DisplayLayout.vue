<script setup lang="ts">
import { computed } from 'vue'
import type { DisplayOrientation, DisplayProfile } from '../theme'
import {
  DISPLAY_THEME_DEFAULTS_DARK,
  DISPLAY_THEME_DEFAULTS_LIGHT,
} from '../theme'

const props = withDefaults(
  defineProps<{
    /** Product name shown in the identity zone; tap targets the launcher. */
    product: string
    /** Persona hint — cosmetic only, products may ignore or extend it. */
    profile?: DisplayProfile
    /** Forces a layout; omit to let CSS `@media (orientation:)` decide. */
    orientation?: DisplayOrientation
    /** Dark is the default for strip displays regardless of host OS theme. */
    theme?: 'dark' | 'light'
  }>(),
  {
    profile: 'sysadmin',
    orientation: 'landscape',
    theme: 'dark',
  },
)

const emit = defineEmits<{
  'open-launcher': []
}>()

const themeVars = computed(() => {
  const vars = props.theme === 'light' ? DISPLAY_THEME_DEFAULTS_LIGHT : DISPLAY_THEME_DEFAULTS_DARK
  return vars as Record<string, string>
})
</script>

<template>
  <div
    class="cf-display-layout"
    :class="[`cf-display-layout--${orientation}`, `cf-display-layout--profile-${profile}`]"
    :style="themeVars"
    :data-cf-display-theme="theme"
  >
    <button
      type="button"
      class="cf-display-layout__identity"
      :aria-label="`${product} — open display launcher`"
      @click="emit('open-launcher')"
    >
      <slot name="identity">
        <span class="cf-display-layout__product">{{ product }}</span>
        <span class="cf-display-layout__profile">{{ profile }}</span>
      </slot>
    </button>

    <div class="cf-display-layout__metrics">
      <slot name="metrics" />
    </div>

    <div class="cf-display-layout__alerts">
      <slot name="alerts" />
    </div>

    <div class="cf-display-layout__macros">
      <slot name="macros" />
    </div>
  </div>
</template>

<style scoped>
.cf-display-layout {
  display: grid;
  width: 100%;
  height: 100vh;
  background: var(--cf-display-surface, #0d1117);
  color: var(--cf-display-text-primary, #e6edf3);
  overflow: hidden;
  font-family: system-ui, sans-serif;
}

/* Landscape: 1920x480 — identity | metrics | alerts | macros, left to right. */
.cf-display-layout--landscape {
  grid-template-columns: 128px minmax(0, 1fr) 300px 192px;
  grid-template-rows: 100%;
  grid-template-areas: 'identity metrics alerts macros';
}

/* Portrait: 480x1920 — stacked top to bottom. */
.cf-display-layout--portrait {
  grid-template-columns: 100%;
  grid-template-rows: 80px minmax(0, 1fr) auto auto;
  grid-template-areas:
    'identity'
    'metrics'
    'alerts'
    'macros';
}

.cf-display-layout__identity {
  grid-area: identity;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.15rem;
  border: none;
  border-right: 1px solid var(--cf-display-surface-border, #30363d);
  background: var(--cf-display-surface-raised, #161b22);
  color: inherit;
  cursor: pointer;
  padding: 0.5rem;
}

.cf-display-layout--portrait .cf-display-layout__identity {
  flex-direction: row;
  border-right: none;
  border-bottom: 1px solid var(--cf-display-surface-border, #30363d);
}

.cf-display-layout__product {
  font-weight: 700;
  font-size: clamp(0.8rem, 2vw, 1.1rem);
}

.cf-display-layout__profile {
  font-size: 0.7rem;
  color: var(--cf-display-text-muted, #8b949e);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.cf-display-layout__metrics {
  grid-area: metrics;
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.5rem;
  overflow: auto;
}

.cf-display-layout--portrait .cf-display-layout__metrics {
  flex-direction: column;
  justify-content: flex-start;
}

.cf-display-layout__alerts {
  grid-area: alerts;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  padding: 0.5rem;
  overflow-y: auto;
  border-left: 1px solid var(--cf-display-surface-border, #30363d);
}

.cf-display-layout--portrait .cf-display-layout__alerts {
  border-left: none;
  border-top: 1px solid var(--cf-display-surface-border, #30363d);
}

.cf-display-layout__macros {
  grid-area: macros;
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 0.4rem;
  align-content: center;
  padding: 0.5rem;
  border-left: 1px solid var(--cf-display-surface-border, #30363d);
}

.cf-display-layout--portrait .cf-display-layout__macros {
  grid-template-columns: repeat(3, 1fr);
  border-left: none;
  border-top: 1px solid var(--cf-display-surface-border, #30363d);
}
</style>
