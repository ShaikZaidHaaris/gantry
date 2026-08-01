import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API runs separately in development; proxying keeps the browser on one
// origin so cookies, SSE and fetch all behave the way they will in production.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 7911,
    proxy: {
      "/api": { target: "http://127.0.0.1:7910", changeOrigin: true },
    },
  },
});
