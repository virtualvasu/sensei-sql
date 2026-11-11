import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { '@': path.resolve(import.meta.dirname, './src') } },
  // `npm run dev` proxies API calls to the Python server (python -m sensei_sql.app)
  server: { proxy: { '/api': 'http://127.0.0.1:7860' } },
})
