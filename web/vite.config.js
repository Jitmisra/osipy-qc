import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Built assets land in web/dist, which the Python server serves directly, so the
// whole app ships as static files with no node runtime in production.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: { outDir: 'dist', emptyOutDir: true },
  server: {
    port: 5173,
    // during development the API still comes from the Python server
    proxy: { '/api': 'http://127.0.0.1:8000', '/run': 'http://127.0.0.1:8000' },
  },
})
