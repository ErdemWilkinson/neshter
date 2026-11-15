import { expect, test } from '@playwright/test';
import { cut, measure, openModel, runPrompt, selectWithKeyboard } from './flow';
import { freshWorkdir, startServer, type RunningServer } from './server';

let server: RunningServer;

test.beforeAll(async () => {
  server = await startServer({ workdir: freshWorkdir('offline') });
});
test.afterAll(async () => {
  await server?.stop();
});

declare global {
  interface Window {
    __csp?: string[];
  }
}

test('every request stays on 127.0.0.1, no CSP violation, nothing in localStorage, bundled fonts', async ({ page }) => {
  await page.addInitScript(() => {
    window.__csp = [];
    document.addEventListener('securitypolicyviolation', (event) => {
      window.__csp?.push(`${event.violatedDirective} ${event.blockedURI}`);
    });
  });
  const urls: string[] = [];
  page.on('request', (request) => urls.push(request.url()));
  const failures: string[] = [];
  page.on('response', (response) => {
    // The restore prompt asks /api/drafts first, so even "no draft yet" must not produce a 404.
    if (response.status() >= 400) {
      failures.push(`${response.status()} ${response.url()}`);
    }
  });
  const pageErrors: string[] = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));

  await page.goto(server.url);
  await openModel(page); // Home → Models → load
  await runPrompt(page); // Explore
  await selectWithKeyboard(page, [[0, 1]]);
  await measure(page);
  await cut(page);

  await page.getByRole('link', { name: 'Davranışı bul' }).click(); // Concept, preset selected
  await page.getByRole('group', { name: 'Kavram' }).locator('label').first().click();
  await expect(page.getByRole('button', { name: 'Ara' })).toBeVisible();

  await page.getByRole('link', { name: 'Karşılaştır', exact: true }).click(); // Compare
  await page.getByLabel('Kendin dene').fill('merhaba');
  await page.getByRole('button', { name: 'İkisine de sor' }).click();
  await expect(page.locator('.pair-side')).toHaveCount(2);

  await page.getByRole('link', { name: 'Kaydet', exact: true }).click(); // Save
  await expect(page.getByLabel('Ad', { exact: true })).toBeVisible();

  const prefix = `http://127.0.0.1:${server.port}/`;
  expect(urls.filter((u) => !u.startsWith(prefix))).toEqual([]);
  expect(urls.length).toBeGreaterThan(10);
  expect(await page.evaluate(() => window.__csp)).toEqual([]);
  expect(await page.evaluate(() => localStorage.length)).toBe(0);
  expect(failures).toEqual([]);
  expect(pageErrors).toEqual([]);
  const fontsOk = await page.evaluate(async () => {
    await document.fonts.ready;
    return document.fonts.check('16px "Instrument Sans Variable"', 'ĞğİıŞş');
  });
  expect(fontsOk).toBe(true);
});
