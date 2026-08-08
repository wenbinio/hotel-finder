import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export function normalizeIndexLineEndings(html) {
  return html.replace(/\r\n?/g, '\n')
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    {
      name: 'normalize-index-line-endings',
      transformIndexHtml: {
        order: 'post',
        handler: normalizeIndexLineEndings,
      },
    },
  ],
  build: {
    manifest: true,
  },
  test: {
    environment: 'jsdom',
    setupFiles: './src/test/setup.js',
  },
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:5001',
    },
  },
})
