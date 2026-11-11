export type OutDirCheck = 'ok' | 'relative' | 'unc' | 'control';

const DRIVE = /^[A-Za-z]:[\\/]/;
// eslint-disable-next-line no-control-regex -- rejecting control characters is the point
const CONTROL = /[\u0000-\u001f\u007f]/;
const BACKSLASH = '\\';

/**
 * The parent folder the user typed. Absolute means a drive letter when the server's own default folder is
 * drive-letter style (on Windows the server rejects `/x`: `Path("/x").is_absolute()` is false there), else a
 * leading `/` (F29). UNC paths are refused.
 */
export function checkOutDir(dir: string, defaultOutDir: string): OutDirCheck {
  if (CONTROL.test(dir)) return 'control';
  if (dir.startsWith(BACKSLASH + BACKSLASH) || dir.startsWith('//')) return 'unc';
  const windows = DRIVE.test(defaultOutDir);
  const absolute = windows ? DRIVE.test(dir) : dir.startsWith('/');
  return absolute ? 'ok' : 'relative';
}

/** `join(dir, name)` with the separator the folder already uses. */
export function joinOutDir(dir: string, name: string): string {
  if (dir.endsWith(BACKSLASH) || dir.endsWith('/')) return `${dir}${name}`;
  return `${dir}${dir.includes(BACKSLASH) ? BACKSLASH : '/'}${name}`;
}
