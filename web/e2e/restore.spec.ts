import { expect, test } from '@playwright/test';
import { cut, measure, openModel, runPrompt, selectWithKeyboard } from './flow';
import { freshWorkdir, startServer } from './server';

test('cuts survive a server restart on another port, and a restart on the same port ends the old tab', async ({
  page,
  context,
}) => {
  const workdir = freshWorkdir('restore');
  const first = await startServer({ workdir });
  const servers = [first];
  try {
    await page.goto(first.url);
    await openModel(page);
    await runPrompt(page);
    await selectWithKeyboard(page, [[0, 1]]);
    await measure(page);
    const saved = page.waitForResponse((r) => r.url().includes('/api/draft/') && r.request().method() === 'PUT');
    await cut(page);
    await expect(page.locator('.chip-unsaved')).toHaveText('Kaydedilmemiş 1 kesim');
    expect((await saved).ok()).toBe(true);

    // The server goes away: the old tab says so (10 s heartbeat; allow for a shared CPU).
    await first.stop();
    await expect(page.getByText('Sunucu kapandı; kesimlerin taslakta')).toBeVisible({ timeout: 25_000 });

    // A new server: other port, other token, same workdir.
    const second = await startServer({ workdir });
    servers.push(second);
    expect(second.port).not.toBe(first.port);
    const fresh = await context.newPage();
    await fresh.goto(second.url);
    await expect(fresh.getByText('Kaydedilmemiş çalışmalar: neshter/tiny-e2e · 1 kesim')).toBeVisible();
    await openModel(fresh);
    await expect(fresh.getByText('Önceki oturumdan kaydedilmemiş 1 kesim var: geri yükle?')).toBeVisible();
    await fresh.getByRole('button', { name: 'Geri yükle' }).click();
    await expect(fresh.locator('.basket-list li')).toHaveCount(1);
    await fresh.close();

    // The first tab polls on: the origin it knows answers again, but with a new token.
    await second.stop();
    const third = await startServer({ workdir, port: first.port });
    servers.push(third);
    await expect(page.getByText('Oturum bitti, sunucu yeniden başlatılmış')).toBeVisible({ timeout: 25_000 });
  } finally {
    await Promise.all(servers.map((s) => s.stop()));
  }
});
