import { expect, test } from '@playwright/test';
import { cut, measure, openModel, runPrompt, selectWithKeyboard } from './flow';
import { freshWorkdir, startServer, type RunningServer } from './server';

let server: RunningServer;

test.beforeAll(async () => {
  server = await startServer({ workdir: freshWorkdir('leave') });
});
test.afterAll(async () => {
  await server?.stop();
});

test('second tab, leave dialog, beforeunload and a clean close', async ({ page, context }) => {
  await page.goto(server.url);

  // The tab lock runs before the token check: the token-stripped URL still says "open in another tab".
  const second = await context.newPage();
  await second.goto(server.origin + '/');
  await expect(second.getByText('Neşter başka bir sekmede açık')).toBeVisible();
  await second.close();

  await openModel(page);
  await runPrompt(page);
  await selectWithKeyboard(page, [[0, 1]]);
  await measure(page);
  await cut(page);
  await expect(page.locator('.chip-unsaved')).toHaveText('Kaydedilmemiş 1 kesim');

  await page.getByRole('button', { name: 'Kapat' }).click();
  const dialog = page.getByRole('dialog');
  await expect(dialog.getByRole('button', { name: 'Kaydet', exact: true })).toBeVisible();
  await expect(dialog.getByRole('button', { name: 'Kaydetmeden çık' })).toBeVisible();
  await dialog.getByRole('button', { name: 'Vazgeç' }).click();
  await expect(dialog).toBeHidden();
  await expect(page.locator('.chip-unsaved')).toBeVisible(); // still in the app

  // beforeunload: dismissed, so the reload never resolves.
  const types: string[] = [];
  page.on('dialog', (d) => {
    types.push(d.type());
    void d.dismiss();
  });
  await Promise.race([page.reload(), new Promise((resolve) => setTimeout(resolve, 3000))]).catch(() => undefined);
  expect(types).toContain('beforeunload');
  await expect(page.locator('.chip-unsaved')).toBeVisible();

  await page.getByRole('button', { name: 'Kapat' }).click();
  await page.getByRole('dialog').getByRole('button', { name: 'Kaydetmeden çık' }).click();
  await expect(page.getByText('Neşter kapandı; bu sekmeyi kapatabilirsiniz')).toBeVisible();
  await Promise.race([
    server.exited,
    new Promise((_, reject) => setTimeout(() => reject(new Error('server still running after 10 s')), 10_000)),
  ]);
  expect(server.child.exitCode).not.toBeNull();
});
