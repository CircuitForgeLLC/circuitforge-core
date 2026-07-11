/**
 * @circuitforge/display — central theme file.
 *
 * Single source of truth for the strip-display design tokens. Components in
 * this package read these as CSS custom properties (with fallbacks, so they
 * render correctly even in a host that hasn't wired the theme up yet).
 *
 * Consuming products that already have a UnoCSS config (Turnstone, Robin)
 * should spread `displayUnoTheme`/`displayUnoShortcuts` into their own
 * `uno.config.ts` so the strip-display route matches their product's theme
 * rather than diverging with a second, unrelated palette. See
 * packages/display/README.md for the merge snippet.
 *
 * Dark theme is the default — strip displays typically sit adjacent to a
 * bright monitor, so a light theme fights for attention. Set
 * `data-cf-display-theme="light"` on the root element to override.
 */

/** CSS custom property names, gathered so components and consumers share one contract. */
export const DISPLAY_THEME_VARS = {
  surface: '--cf-display-surface',
  surfaceRaised: '--cf-display-surface-raised',
  surfaceBorder: '--cf-display-surface-border',
  accent: '--cf-display-accent',
  accentMuted: '--cf-display-accent-muted',
  textPrimary: '--cf-display-text-primary',
  textMuted: '--cf-display-text-muted',
  textDim: '--cf-display-text-dim',
  sevOk: '--cf-display-sev-ok',
  sevWarn: '--cf-display-sev-warn',
  sevCrit: '--cf-display-sev-crit',
} as const

/** Default values, dark theme. Applied as :root fallbacks by DisplayLayout. */
export const DISPLAY_THEME_DEFAULTS_DARK: Record<string, string> = {
  [DISPLAY_THEME_VARS.surface]: '#0d1117',
  [DISPLAY_THEME_VARS.surfaceRaised]: '#161b22',
  [DISPLAY_THEME_VARS.surfaceBorder]: '#30363d',
  [DISPLAY_THEME_VARS.accent]: '#39d353',
  [DISPLAY_THEME_VARS.accentMuted]: '#1f6feb',
  [DISPLAY_THEME_VARS.textPrimary]: '#e6edf3',
  [DISPLAY_THEME_VARS.textMuted]: '#8b949e',
  [DISPLAY_THEME_VARS.textDim]: '#484f58',
  [DISPLAY_THEME_VARS.sevOk]: '#3fb950',
  [DISPLAY_THEME_VARS.sevWarn]: '#d29922',
  [DISPLAY_THEME_VARS.sevCrit]: '#f85149',
}

/** Light theme override values, applied when data-cf-display-theme="light". */
export const DISPLAY_THEME_DEFAULTS_LIGHT: Record<string, string> = {
  [DISPLAY_THEME_VARS.surface]: '#ffffff',
  [DISPLAY_THEME_VARS.surfaceRaised]: '#f6f8fa',
  [DISPLAY_THEME_VARS.surfaceBorder]: '#d0d7de',
  [DISPLAY_THEME_VARS.accent]: '#1f883d',
  [DISPLAY_THEME_VARS.accentMuted]: '#0969da',
  [DISPLAY_THEME_VARS.textPrimary]: '#1f2328',
  [DISPLAY_THEME_VARS.textMuted]: '#59636e',
  [DISPLAY_THEME_VARS.textDim]: '#8c959f',
  [DISPLAY_THEME_VARS.sevOk]: '#1a7f37',
  [DISPLAY_THEME_VARS.sevWarn]: '#9a6700',
  [DISPLAY_THEME_VARS.sevCrit]: '#cf222e',
}

export type DisplayProfile = 'sysadmin' | 'gamer' | 'pro' | 'casual'
export type DisplayOrientation = 'landscape' | 'portrait'
export type DisplaySeverity = 'ok' | 'warn' | 'crit'

/** Cosmetic-only hints per profile — products may ignore or extend these. */
export const DISPLAY_PROFILE_DENSITY: Record<DisplayProfile, 'dense' | 'bold' | 'clean' | 'spacious'> = {
  sysadmin: 'dense',
  gamer: 'bold',
  pro: 'clean',
  casual: 'spacious',
}

/**
 * UnoCSS theme extension fragment. Spread into a consuming product's own
 * `uno.config.ts` theme block (`theme: { colors: { ...displayUnoTheme.colors } }`)
 * so `uno-*` utility classes referencing these tokens are available.
 */
export const displayUnoTheme = {
  colors: {
    cfDisplay: {
      surface: `var(${DISPLAY_THEME_VARS.surface})`,
      surfaceRaised: `var(${DISPLAY_THEME_VARS.surfaceRaised})`,
      surfaceBorder: `var(${DISPLAY_THEME_VARS.surfaceBorder})`,
      accent: `var(${DISPLAY_THEME_VARS.accent})`,
      accentMuted: `var(${DISPLAY_THEME_VARS.accentMuted})`,
      textPrimary: `var(${DISPLAY_THEME_VARS.textPrimary})`,
      textMuted: `var(${DISPLAY_THEME_VARS.textMuted})`,
      textDim: `var(${DISPLAY_THEME_VARS.textDim})`,
      sevOk: `var(${DISPLAY_THEME_VARS.sevOk})`,
      sevWarn: `var(${DISPLAY_THEME_VARS.sevWarn})`,
      sevCrit: `var(${DISPLAY_THEME_VARS.sevCrit})`,
    },
  },
}

/**
 * UnoCSS shortcuts fragment, per the strip display spec's "two additions
 * needed": display-metric, display-macro, display-alert. Spread into
 * `shortcuts: { ...displayUnoShortcuts }`.
 */
export const displayUnoShortcuts: Record<string, string> = {
  'display-metric': 'flex flex-col items-center justify-center rounded-md px-3 py-2 min-w-24',
  // 44px is the accessibility-minimum touch target; strip displays prefer 64px+.
  'display-macro': 'flex flex-col items-center justify-center rounded-lg min-h-16 min-w-16 select-none',
  'display-alert': 'flex items-center gap-2 border-l-4 px-2 py-1 text-sm truncate',
}
