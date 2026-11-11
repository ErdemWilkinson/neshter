import { formatOptions, ollamaAvailable } from '../formats';
import { checkName } from '../nameRule';
import { checkOutDir, joinOutDir } from '../outDir';

test.each([
  ['qwen2.5-0.5b-neshter', 'ok'],
  ['şiir', 'chars'],
  ['CON', 'reserved'],
  ['nul.txt', 'reserved'],
  ['com3', 'reserved'],
  ['model.', 'trailing'],
  ['model ', 'trailing'],
  ['', 'empty'],
  ['-x', 'chars'],
  ['a'.repeat(65), 'chars'],
  ['a'.repeat(64), 'ok'],
  ['console', 'ok'],
])('checkName %s', (name, expected) => expect(checkName(name)).toBe(expected));

const WIN = 'C:\\Users\\x\\Neshter';
test.each([
  ['C:\\Modeller', WIN, 'ok'],
  ['D:/Modeller', WIN, 'ok'],
  ['modeller', WIN, 'relative'],
  ['/x', WIN, 'relative'], // F29: the server rejects `/x` on Windows
  ['', WIN, 'relative'],
  ['/home/x/m', '/home/x/Neshter', 'ok'],
  ['C:\\x', '/home/x/Neshter', 'relative'],
  ['\\\\srv\\x', WIN, 'unc'],
  ['//srv/x', '/home/x/Neshter', 'unc'],
  ['C:\\a\nb', WIN, 'control'],
])('checkOutDir %s', (dir, defaultDir, expected) => expect(checkOutDir(dir, defaultDir)).toBe(expected));

test.each([
  ['C:\\Modeller', 'kesik', 'C:\\Modeller\\kesik'],
  ['C:\\Modeller\\', 'kesik', 'C:\\Modeller\\kesik'],
  ['D:/m', 'kesik', 'D:/m/kesik'],
  ['/home/x', 'kesik', '/home/x/kesik'],
])('joinOutDir %s', (dir, name, expected) => expect(joinOutDir(dir, name)).toBe(expected));

const llama = (found: boolean, converter: boolean) => ({ llama_cpp: { found, path: null, converter } });
test('formatOptions: GGUF needs llama.cpp and its converter, then it is experimental', () => {
  expect(formatOptions(llama(false, false)).map((o) => [o.id, o.enabled, o.note])).toEqual([
    ['safetensors', true, null],
    ['gguf_q4_k_m', false, 'needs_llama'],
    ['gguf_q8_0', false, 'needs_llama'],
  ]);
  expect(formatOptions(llama(true, false))[1]?.enabled).toBe(false);
  expect(formatOptions(llama(true, true)).map((o) => [o.enabled, o.note])).toEqual([
    [true, null],
    [true, 'experimental'],
    [true, 'experimental'],
  ]);
});

test('ollamaAvailable: GGUF and ollama on PATH only', () => {
  expect(ollamaAvailable({ ollama: true }, 'safetensors')).toBe(false);
  expect(ollamaAvailable({ ollama: true }, 'gguf_q8_0')).toBe(true);
  expect(ollamaAvailable({ ollama: false }, 'gguf_q8_0')).toBe(false);
  expect(ollamaAvailable(null, 'gguf_q8_0')).toBe(false);
});
