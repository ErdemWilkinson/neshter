import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { jobMessage } from '../jobMessage';

const ROOT = resolve(__dirname, '../../../..');

function pyFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) return name === '__pycache__' ? [] : pyFiles(path);
    return name.endsWith('.py') ? [path] : [];
  });
}

/** Every literal message the server passes to `progress(...)` / `report(...)`, f-string fields filled with 3. */
function serverMessages(): string[] {
  const found = new Set<string>();
  const call = /\b(?:progress|report|head_progress)\(\s*[^,()]*(?:\([^()]*\)[^,()]*)*,\s*(f?)"((?:[^"\\]|\\.)*)"/g;
  for (const file of pyFiles(join(ROOT, 'neshter'))) {
    for (const m of readFileSync(file, 'utf8').matchAll(call)) {
      const text = m[1] === 'f' ? (m[2] ?? '').replace(/\{[^}]*\}/g, '3') : (m[2] ?? '');
      found.add(text);
    }
  }
  // The MB counter is split over two lines in app.py.
  found.add('3 / 3 MB indirildi');
  return [...found];
}

function deviceMessages(): string[] {
  const source = readFileSync(join(ROOT, 'web/src/api/device/deviceApi.ts'), 'utf8');
  return [...source.matchAll(/message: '([^']+)'/g)].map((m) => m[1] ?? '');
}

test('the scan finds the server messages', () => {
  const messages = serverMessages();
  expect(messages.length).toBeGreaterThan(30);
  expect(messages).toContain('Rastgele kontrol 3/3 (erken durabilir)');
});

test.each([...serverMessages(), ...deviceMessages()])('the English UI translates %j', (message) => {
  const english = jobMessage(message, 'en');
  expect(english).not.toBe(message);
  expect(english).not.toMatch(/[çğıöşüÇĞİÖŞÜ]/);
});

test('the Turkish UI shows the message as it is, and unknown messages pass through', () => {
  expect(jobMessage('Rastgele kontrol 3/39', 'tr')).toBe('Rastgele kontrol 3/39');
  expect(jobMessage('Rastgele kontrol 3/39 (erken durabilir)', 'en')).toBe('Random control 3/39 (may stop early)');
  expect(jobMessage('something new', 'en')).toBe('something new');
  expect(jobMessage('', 'en')).toBe('');
});
