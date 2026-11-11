import { fileURLToPath } from 'node:url';
import react from '@vitejs/plugin-react';
import { searchForWorkspaceRoot } from 'vite';
import { defineConfig } from 'vitest/config';

const here = fileURLToPath(new URL('.', import.meta.url));
const fontFiles = (pkg: string) => fileURLToPath(new URL(`./node_modules/@fontsource-variable/${pkg}/files`, import.meta.url));

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@fonts/inter': fontFiles('inter'),
      '@fonts/space-grotesk': fontFiles('space-grotesk'),
      '@fonts/jetbrains-mono': fontFiles('jetbrains-mono'),
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  build: {
    outDir: '../neshter/web/dist',
    emptyOutDir: true,
    assetsInlineLimit: 0,
    modulePreload: { polyfill: false },
  },
  server: {
    // web/node_modules is a junction to D:; Vite serves its files from the realpath.
    fs: { allow: [searchForWorkspaceRoot(here), 'D:/neshter/web/node_modules'] },
    proxy: {
      '/api': {
        target: `http://127.0.0.1:${process.env['NESHTER_DEV_PORT'] ?? '8765'}`,
        changeOrigin: true,
        configure: (proxy) => {
          proxy.on('proxyReq', (proxyReq) => {
            proxyReq.removeHeader('origin');
            const token = process.env['NESHTER_DEV_TOKEN'];
            if (token) proxyReq.setHeader('x-neshter-token', token);
          });
        },
      },
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./vitest.setup.ts'],
    include: ['src/**/*.test.{ts,tsx}', 'scripts/__tests__/*.test.ts'],
  },
});
