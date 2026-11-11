import { plural, resolveLang, setLang, t } from '..';
import en from '../en.json';
import tr from '../tr.json';

afterEach(() => setLang('tr'));

test('t interpolates and formats numbers per language', () => {
  setLang('tr');
  expect(t('header.vram', { gb: 0.025 })).toBe('VRAM 0,025 GB');
  expect(t('header.model', { name: 'qwen {x}', dtype: 'float32', device: 'cpu' })).toBe('qwen {x} · float32 · cpu');
  setLang('en');
  expect(t('header.vram', { gb: 0.025 })).toBe('VRAM 0.025 GB');
  expect(t('header.close')).toBe('Quit Neşter');
});

test('a placeholder without a parameter stays visible instead of vanishing', () => {
  expect(t('header.ram')).toBe('RAM {gb} GB');
  expect(t('header.ram', {})).toBe('RAM {gb} GB');
});

test('setLang sets html lang and the store', () => {
  setLang('en');
  expect(document.documentElement.lang).toBe('en');
  setLang('tr');
  expect(document.documentElement.lang).toBe('tr');
});

test('default language follows navigator when prefs.lang is null', () => {
  expect(resolveLang(null, 'tr-TR')).toBe('tr');
  expect(resolveLang(null, 'TR')).toBe('tr');
  expect(resolveLang(null, 'en-US')).toBe('en');
  expect(resolveLang(null, 'de')).toBe('en');
  expect(resolveLang(null, undefined)).toBe('en');
  expect(resolveLang('en', 'tr-TR')).toBe('en');
  expect(resolveLang('tr', 'en-US')).toBe('tr');
});

test('plural picks the singular form where the language has one', () => {
  setLang('en');
  expect(plural('header.unsaved', 1)).toBe('1 unsaved cut');
  expect(plural('header.unsaved', 2)).toBe('2 unsaved cuts');
  setLang('tr');
  expect(plural('header.unsaved', 1)).toBe('Kaydedilmemiş 1 kesim');
  expect(plural('header.unsaved', 2)).toBe('Kaydedilmemiş 2 kesim');
});

test('exact pinned copy (Task 7 keys only)', () => {
  setLang('tr');
  expect(t('header.unsaved', { n: 2 })).toBe('Kaydedilmemiş 2 kesim');
  expect(t('app.stale.restarted')).toBe('Oturum bitti, sunucu yeniden başlatılmış');
  expect(t('app.stale.down')).toBe('Sunucu kapandı; kesimlerin taslakta');
  expect(t('header.cpuOnly')).toBe('Yalnızca CPU');
  expect(t('header.motion')).toBe('Animasyonları azalt');
  expect(t('header.close')).toBe("Neşter'i kapat");
  expect(t('ui.techDetail')).toBe('Teknik ayrıntı');
  expect(t('ui.cancel')).toBe('İptal');
  expect(t('app.step.concept')).toBe('Davranışı bul');
  expect(t('app.step.explore')).toBe('Ağ haritası');
  setLang('en');
  expect(t('header.unsaved', { n: 2 })).toBe('2 unsaved cuts');
});

test('forbidden copy is absent from catalogs', () => {
  const all = JSON.stringify(tr) + JSON.stringify(en);
  for (const s of [
    'ortalama değeriyle',
    'Kaynağa güveniyorum',
    'Güvenli formata dönüştür',
    'katmanları sırayla yükler',
    'MLP nöronları',
    'Kendi test setini yükle',
    'Modeli küçült',
    'Dikkat paneli',
    'attention panel',
  ]) {
    expect(all).not.toContain(s);
  }
  expect(all).not.toMatch(/~\s*\d+\s*(dk|sn|min)/); // invented ETAs
  expect(all).not.toMatch(/\d+\s*dk\b/);
  expect(all).not.toMatch(/rastgele[^"]*oran/i); // rate-level random-control line
});

test('every catalog value is non-empty and both catalogs have the same keys', () => {
  expect(Object.keys(en).sort()).toEqual(Object.keys(tr).sort());
  for (const value of [...Object.values(tr), ...Object.values(en)]) expect(value.trim()).not.toBe('');
});
