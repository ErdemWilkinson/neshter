import { expect, type Locator, type Page } from '@playwright/test';

export const MODEL_NAME = 'neshter/tiny-e2e';

/** Home → `İçine bak` → the tiny model's row → `Aç` → the Explore screen. */
export async function openModel(page: Page, intent: RegExp = /İçine bak/): Promise<void> {
  await page.getByRole('button', { name: intent }).click();
  const row = page.locator('.model-row', { hasText: MODEL_NAME });
  await expect(row).toBeVisible();
  await row.getByRole('button', { name: 'Aç' }).click();
}

export async function runPrompt(page: Page, text = 'merhaba dünya'): Promise<void> {
  await page.getByLabel('Metin').fill(text);
  await page.getByRole('button', { name: 'Çalıştır' }).click();
  await expect(page.locator('canvas.heatmap-canvas')).toBeVisible();
}

export function heatmap(page: Page): Locator {
  return page.locator('canvas.heatmap-canvas');
}

/** Focuses the heatmap and selects the given (layer, head) cells with arrows and Space only. */
export async function selectWithKeyboard(page: Page, cells: readonly [number, number][]): Promise<void> {
  const canvas = heatmap(page);
  await canvas.focus(); // lands on K0 H0
  let at: [number, number] = [0, 0];
  for (const [layer, head] of cells) {
    while (at[0] < layer) {
      await page.keyboard.press('ArrowDown');
      at = [at[0] + 1, at[1]];
    }
    while (at[1] < head) {
      await page.keyboard.press('ArrowRight');
      at = [at[0], at[1] + 1];
    }
    while (at[1] > head) {
      await page.keyboard.press('ArrowLeft');
      at = [at[0], at[1] - 1];
    }
    await page.keyboard.press('Space');
  }
}

export async function measure(page: Page): Promise<void> {
  await page.getByRole('button', { name: 'Gerçek etkiyi ölç' }).click();
  await expect(page.locator('.control-card')).toContainText(/daha etkili|ayırt edilemedi/, { timeout: 120_000 });
}

export async function cut(page: Page): Promise<void> {
  await page.getByRole('button', { name: /Kes \(geçici\)/ }).click();
}
