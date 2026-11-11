import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: 'e2e',
  outputDir: 'node_modules/.pw-results',
  workers: 1,
  fullyParallel: false,
  reporter: 'list',
  timeout: 180_000,
  expect: { timeout: 20_000 },
  use: { browserName: 'chromium', headless: true, trace: 'off', locale: 'tr-TR' },
  projects: [{ name: 'chromium' }],
});
