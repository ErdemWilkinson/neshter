import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { expect, test } from '@playwright/test';
import { cut, heatmap, measure, openModel, runPrompt, selectWithKeyboard } from './flow';
import { freshWorkdir, startServer, type RunningServer } from './server';

let server: RunningServer;
let workdir: string;

test.beforeAll(async () => {
  workdir = freshWorkdir('happy');
  server = await startServer({ workdir });
});
test.afterAll(async () => {
  await server?.stop();
});

test('Home → Explore → measure → cut → Compare → Save', async ({ page }) => {
  await page.goto(server.url);
  await openModel(page);
  await runPrompt(page);

  // 2D heatmap: keyboard only (Space toggles the focused cell).
  await selectWithKeyboard(page, [
    [0, 1],
    [1, 2],
  ]);
  const chips = page.getByRole('list', { name: "Seçili head'ler" });
  await expect(chips).toContainText('K0·H1');
  await expect(chips).toContainText('K1·H2');
  await expect(heatmap(page)).toBeVisible();

  // Liste view: the tiny model has 2 layers × 4 heads.
  await page.getByLabel('Liste', { exact: true }).check();
  await expect(page.locator('table.heatmap-table tbody tr')).toHaveCount(8);
  await page.getByLabel('2B ısı haritası').check();

  await measure(page);
  await cut(page);
  await expect(page.locator('.basket-button.has-unsaved')).toHaveText('Kaydedilmemiş 2 kesim');

  await page.getByRole('button', { name: 'Karşılaştır' }).click();
  await page.getByLabel('Kendin dene').fill('merhaba dünya');
  await page.getByRole('button', { name: 'İkisine de sor' }).click();
  await expect(page.locator('.pair-side')).toHaveCount(2);
  await expect(page.locator('.pair-side').first().getByRole('heading', { name: 'Orijinal' })).toBeVisible();

  // Heatmap cuts carry no concept: no target category, so the fallback sentence and the general counts.
  await page.getByRole('button', { name: 'Test setini koş' }).click();
  await expect(page.getByText('Bu kavramın test kategorisi yok; aşağıdaki genel sayılara bakın.')).toBeVisible({
    timeout: 170_000,
  });
  await expect(page.locator('table.suite-table').getByRole('row', { name: /Genel/ })).toBeVisible();

  // Save.
  await page.getByRole('link', { name: 'Kaydet', exact: true }).click();
  await page.getByLabel('Ad', { exact: true }).fill('tiny-e2e-neshter');
  await expect(page.getByLabel('safetensors')).toBeChecked();
  await page.getByRole('button', { name: /^Kaydet/ }).click();

  const card = page.locator('.result-card');
  await expect(card.getByRole('heading', { name: 'Kaydedildi' })).toBeVisible({ timeout: 170_000 });
  await expect(card.locator('.result-sha')).toHaveText(/^[0-9a-f]{64}$/);
  await expect(card).toContainText('Orijinal dosyaya dokunulmadı');
  await expect(page.locator('.basket-button.has-unsaved')).toHaveCount(0);

  const out = join(workdir, 'home', 'tiny-e2e-neshter');
  expect(existsSync(out)).toBe(true);
});
