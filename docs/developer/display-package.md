# @circuitforge/display — Vue package

`packages/display/` is the first non-Python module in circuitforge-core: Vue 3 primitives for CircuitForge products running a secondary strip display (1920×480 landscape / 480×1920 portrait kiosk). Design spec: `circuitforge-plans/circuitforge-core/superpowers/specs/2026-05-17-strip-display-spec.md`.

## Why a separate npm package, not `circuitforge_core`

Most products in the menagerie are Python-only and never touch Vue. Publishing `@circuitforge/display` as its own npm package — rather than bundling it into the Python `circuitforge-core` distribution — means:

- Products that don't use a strip display (most of them) never pull in Vue as a transitive dependency.
- Versioning follows npm semver independently of the Python package's release cadence — a Vue component API change doesn't force a `circuitforge-core` PyPI bump, and vice versa.
- The build tooling (Vite, vue-tsc, Vitest) stays isolated in `packages/display/`, not mixed into the Python `pyproject.toml`/pytest setup.

This mirrors the reasoning behind keeping the BSL `resources` module split into the separate `circuitforge-orch` package — different consumers, different release cycles, kept apart even though both live in adjacent repos/directories.

## Location and consumers

- Package root: `packages/display/`
- First consumer: Turnstone's sysadmin profile (`turnstone#25`) — CPU/RAM/disk/network metrics, live alert feed, service macro buttons
- Planned: Robin's proactive assistant overlay

## Working on it

```bash
cd packages/display
npm install
npm test          # Vitest — 37 tests across the four components
npm run build     # vue-tsc --emitDeclarationOnly && vite build → dist/
```

Gotcha to watch for: Vite's `build.emptyOutDir` defaults to `true` and will silently wipe the `.d.ts` files `vue-tsc --emitDeclarationOnly` just wrote, since both commands target the same `dist/`. `vite.config.ts` sets `emptyOutDir: false` to prevent this — don't remove it without changing the build order.

See `packages/display/README.md` for the component API, theming (`src/theme.ts` is the central theme file — CSS custom properties + UnoCSS fragments for products that already run UnoCSS), and the launcher page.
