import { resolve } from 'node:path'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  build: {
    // vue-tsc --emitDeclarationOnly runs before this and writes .d.ts files
    // into dist/ — don't let Vite's default emptyOutDir wipe them out.
    emptyOutDir: false,
    lib: {
      entry: resolve(__dirname, 'src/index.ts'),
      name: 'CircuitForgeDisplay',
      fileName: 'circuitforge-display',
      formats: ['es', 'cjs'],
    },
    rollupOptions: {
      external: ['vue'],
    },
  },
  test: {
    environment: 'jsdom',
    include: ['src/tests/**/*.spec.ts'],
  },
})
