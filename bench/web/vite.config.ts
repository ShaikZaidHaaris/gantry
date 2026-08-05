import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API runs separately in development; proxying keeps the browser on one
// origin so cookies, SSE and fetch all behave the way they will in production.
//
// Both ports come from the environment because two people (or two sessions)
// working in this repo at once would otherwise fight over one pair of hardcoded
// numbers -- and the failure is confusing rather than obvious: the second
// server silently loses the bind, and the browser talks to the first one's API
// while showing the second one's code.
const api = process.env.BENCH_API ?? "http://127.0.0.1:7910";

export default defineConfig({
  plugins: [react()],
  server: {
    port: Number(process.env.BENCH_WEB_PORT ?? 7911),
    strictPort: true,
    proxy: {
      "/api": { target: api, changeOrigin: true },
    },
  },
});
