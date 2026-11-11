import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ModelEntry } from '@/api/schemas';
import models from '@/api/__fixtures__/models.json';
import { setLang } from '@/i18n';
import { badgeLabel, formatArch, formatBytes } from '../format';
import { ModelRow } from '../ModelRow';

const base = models[0] as unknown as ModelEntry;
const row = (patch: Partial<ModelEntry> = {}): ModelEntry => ({ ...base, ...patch });

function renderRow(patch: Partial<ModelEntry> = {}, onOpen = vi.fn()) {
  render(
    <ul>
      <ModelRow row={row(patch)} onOpen={onOpen} />
    </ul>,
  );
  return onOpen;
}

beforeEach(() => setLang('tr'));

test('formatArch uses the language number format', () => {
  setLang('tr');
  const mlp = new Intl.NumberFormat('tr').format(4864);
  expect(formatArch({ family: 'qwen2', n_layers: 24, n_heads: 14, n_kv_heads: 2, d_mlp: 4864 })).toBe(
    `24 katman · 14Q/2KV head · ${mlp} MLP`,
  );
  setLang('en');
  expect(formatArch({ family: 'qwen2', n_layers: 24, n_heads: 14, n_kv_heads: 2, d_mlp: 4864 })).toBe(
    `24 layers · 14Q/2KV heads · ${new Intl.NumberFormat('en').format(4864)} MLP`,
  );
});

test('formatArch shows ? for counts the header could not give', () => {
  expect(formatArch({ family: 'qwen2', n_layers: null, n_heads: 4, n_kv_heads: null, d_mlp: 128 })).toBe(
    '? katman · 4Q/?KV head · 128 MLP',
  );
});

test('formatBytes picks a unit', () => {
  expect(formatBytes(512)).toBe('512 B');
  expect(formatBytes(562104)).toBe('549 KB');
  expect(formatBytes(3 * 1024 ** 3)).toBe(`${new Intl.NumberFormat('tr', { maximumFractionDigits: 1 }).format(3)} GB`);
});

test('badge labels', () => {
  expect(badgeLabel('ok')).toBe('rahat');
  expect(badgeLabel('slow')).toBe('yavaş · RAM desteği');
  expect(badgeLabel('no')).toBe('sığmaz');
});

test('a row shows its facts and a null license as unknown', () => {
  renderRow({ license: null });
  expect(screen.getByText('tiny/qwen2')).toBeInTheDocument();
  expect(screen.getByText(/huggingface · safetensors/)).toBeInTheDocument();
  expect(screen.getByText('2 katman · 4Q/2KV head · 128 MLP')).toBeInTheDocument();
  expect(screen.getByText(/lisans bilinmiyor/)).toBeInTheDocument();
  expect(screen.getByText('yavaş · RAM desteği')).toBeInTheDocument();
});

test('a supported row opens through the callback', async () => {
  const onOpen = renderRow();
  await userEvent.click(screen.getByRole('button', { name: 'Aç' }));
  expect(onOpen).toHaveBeenCalledWith(expect.objectContaining({ name: 'tiny/qwen2' }));
});

test.each([
  ['unknown_arch', 'Model mimarisi okunamadı.'],
  ['unsupported_family', 'Bu model ailesi henüz desteklenmiyor.'],
])('unsupported row is disabled with its reason (%s)', async (code, text) => {
  const onOpen = renderRow({ supported: false, unsupported_reason: code, arch: null });
  expect(screen.getByRole('button', { name: 'Aç' })).toBeDisabled();
  expect(screen.getByText(text)).toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: 'Aç' }));
  expect(onOpen).not.toHaveBeenCalled();
});

test('model name with markup renders as text', () => {
  renderRow({ name: '<img src=x onerror=alert(1)>', source: '<b>x</b>', license: '<i>mit</i>' });
  expect(document.querySelector('img')).toBeNull();
  expect(document.querySelector('b')).toBeNull();
  expect(document.querySelector('i')).toBeNull();
  expect(screen.getByText('<img src=x onerror=alert(1)>')).toBeTruthy();
});
