export type NameCheck = 'ok' | 'empty' | 'chars' | 'reserved' | 'trailing';

const VALID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;
const RESERVED = /^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$/i;

/** The export folder name: Windows-safe, ASCII only (the same rule as the server's name check). */
export function checkName(name: string): NameCheck {
  if (name === '') return 'empty';
  if (name.endsWith('.') || name.endsWith(' ')) return 'trailing';
  if (!VALID.test(name)) return 'chars';
  // "nul.txt" is reserved too: Windows looks at the part before the first dot.
  if (RESERVED.test(name.split('.')[0] ?? '')) return 'reserved';
  return 'ok';
}
