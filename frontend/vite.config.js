import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    allowedHosts: ['project-aegis-y--m3g.fly.dev']
  },
  preview: {
    host: '0.0.0.0',
    port: 5173,
    allowedHosts: ['project-aegis-y--m3g.fly.dev']
  }
})
