import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Page } from '@playwright/test';
import { cut, heatmap, measure, openModel, runPrompt, selectWithKeyboard } from './flow';
import { freshWorkdir, startServer, type RunningServer } from './server';

let server: RunningServer;

test.beforeAll(async () => {
  server = await startServer({ workdir: freshWorkdir('a11y') });
});
test.afterAll(async () => {
  await server?.stop();
});

async function expectNoSeriousViolations(page: Page, where: string): Promise<void> {
  // @axe-core/playwright resolves its own playwright-core copy; the runtime object is the same Page.
  const { violations, passes } = await new AxeBuilder({ page: page as unknown as ConstructorParameters<typeof AxeBuilder>[0]['page'] }).analyze();
  expect(passes.length, `${where}: axe ran no rules`).toBeGreaterThan(20);
  const serious = violations.filter((v) => v.impact === 'serious' || v.impact === 'critical');
  expect(
    serious.map((v) => `${where}: ${v.id} (${v.impact}) ${v.nodes.map((n) => n.target.join(' ')).join(' | ')}`),
  ).toEqual([]);
}

test('axe: no serious or critical violations on any screen', async ({ page }) => {
  await page.goto(server.url);
  await expect(page.getByRole('heading', { name: 'Ne yapmak istiyorsun?' })).toBeVisible();
  await expectNoSeriousViolations(page, 'Home');

  await page.getByRole('button', { name: /İçine bak/ }).click();
  await expect(page.locator('.model-row')).toBeVisible();
  await expectNoSeriousViolations(page, 'Models');

  await page.locator('.model-row').getByRole('button', { name: 'Aç' }).click();
  await runPrompt(page);
  await selectWithKeyboard(page, [[0, 1]]);
  await measure(page);
  await expectNoSeriousViolations(page, 'Explore 2D');

  await page.getByLabel('Liste', { exact: true }).check();
  await expect(page.locator('table.heatmap-table')).toBeVisible();
  await expectNoSeriousViolations(page, 'Explore Liste');

  await cut(page);
  await page.getByRole('link', { name: 'Davranışı bul' }).click();
  await expect(page.getByRole('button', { name: 'Ara' })).toBeVisible();
  await expectNoSeriousViolations(page, 'Concept');

  await page.getByRole('link', { name: 'Karşılaştır', exact: true }).click();
  await page.getByLabel('Kendin dene').fill('merhaba');
  await page.getByRole('button', { name: 'İkisine de sor' }).click();
  await expect(page.locator('.pair-side')).toHaveCount(2);
  await expectNoSeriousViolations(page, 'Compare');

  await page.getByRole('link', { name: 'Kaydet', exact: true }).click();
  await expect(page.getByLabel('Ad', { exact: true })).toBeVisible();
  await expectNoSeriousViolations(page, 'Save');
});

test('Liste parity: select, measure and cut without the canvas', async ({ page }) => {
  await page.goto(server.url);
  await openModel(page);
  await runPrompt(page);
  await page.getByLabel('Liste', { exact: true }).check();
  await page.getByLabel('K1 · H3 seç').check();
  await expect(page.getByRole('list', { name: "Seçili head'ler" })).toContainText('K1·H3');
  await measure(page);
  await cut(page);
  await expect(page.locator('.basket-list li')).toHaveCount(1);
  await expect(page.locator('.chip-unsaved')).toHaveText('Kaydedilmemiş 1 kesim');
});

test('reduced motion and a visible focus ring on the heatmap', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto(server.url);
  await expect(page.locator('html[data-motion="reduce"]')).toHaveCount(1);

  await openModel(page);
  await runPrompt(page);
  await heatmap(page).focus();
  await page.keyboard.press('ArrowRight');
  const ring = await heatmap(page).evaluate((el) => {
    const style = getComputedStyle(el);
    return { style: style.outlineStyle, width: parseFloat(style.outlineWidth) };
  });
  expect(ring.style).not.toBe('none');
  expect(ring.width).toBeGreaterThan(0);
});
