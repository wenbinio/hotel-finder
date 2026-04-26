import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Build into ../static so Flask serves the bundle directly.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../static",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:5001",
    },
  },
});
