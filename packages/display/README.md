# @circuitforge/display

Vue 3 primitives for CircuitForge products that run on or alongside a secondary strip display (8.8" USB-C touch, 1920×480 landscape / 480×1920 portrait). Design spec: `circuitforge-plans/circuitforge-core/superpowers/specs/2026-05-17-strip-display-spec.md`.

Published as a separate npm package (not part of the Python `circuitforge-core` distribution) so products that don't use it never pull in Vue as a dependency.

## Install

```bash
npm install @circuitforge/display
```

`vue@^3.5` is a peer dependency.

## Usage

```vue
<script setup lang="ts">
import { DisplayLayout, DisplayMetric, DisplayAlert, DisplayMacroButton } from '@circuitforge/display'
import '@circuitforge/display/style.css'
</script>

<template>
  <DisplayLayout product="turnstone" profile="sysadmin" orientation="landscape">
    <template #metrics>
      <DisplayMetric :value="87" unit="°C" label="CPU temp" severity="warn" />
      <DisplayMetric :value="42" unit="%" label="RAM" />
    </template>
    <template #alerts>
      <DisplayAlert
        message="pacman lock detected — another process is using the database"
        timestamp="2026-01-01T14:22:00"
        severity="crit"
      />
    </template>
    <template #macros>
      <DisplayMacroButton
        icon="🔄"
        label="Restart nginx"
        :action="{ type: 'shell', command: 'systemctl restart nginx' }"
        @trigger="handleMacro"
      />
    </template>
  </DisplayLayout>
</template>
```

A product's `/display` route wraps its content in `DisplayLayout`, filling the `#metrics`, `#alerts`, and `#macros` slots — it doesn't manage zones/orientation manually. `?profile=` and orientation are read from the route by the consuming product and passed in as props (this package doesn't read `window.location` itself, for SSR-safety and testability).

## Components

| Component | Purpose |
|---|---|
| `DisplayLayout` | Root layout — identity zone, orientation-aware grid (landscape/portrait), theme. |
| `DisplayMetric` | Single metric tile — value, label, optional unit/sparkline, severity colour. |
| `DisplayAlert` | Single alert row — timestamp, message, severity-coloured left border. |
| `DisplayMacroButton` | Large touch target (44px+ min) firing a `shell` / `url` / `api` / `display_switch` action. |

## Theming

`src/theme.ts` is the central theme file — CSS custom properties (`--cf-display-*`) with dark-theme defaults, plus `displayUnoTheme`/`displayUnoShortcuts` fragments for products that already run UnoCSS (Turnstone, Robin) to spread into their own `uno.config.ts`:

```ts
// uno.config.ts
import { defineConfig } from 'unocss'
import { displayUnoTheme, displayUnoShortcuts } from '@circuitforge/display'

export default defineConfig({
  theme: { colors: { ...displayUnoTheme.colors } },
  shortcuts: { ...displayUnoShortcuts },
})
```

Dark is the default (strip displays typically sit adjacent to a bright monitor). Pass `theme="light"` to `DisplayLayout` to override.

## Launcher

`launcher/launcher.html` is a static, framework-free page listing configured product display URLs (read from a sibling `launcher.config.json`) and letting the user tap to switch the kiosk window between them. Not part of the npm package's JS API — copy it into a product's static assets or serve it directly.

## What this package does *not* do

No live metrics transport (WebSocket/SSE) — that's product-side (`DisplayDataProvider`, per the spec, lives in the consuming product). No macro execution — `DisplayMacroButton` only emits a `trigger` event with the action payload; the consuming product's backend runs it.
