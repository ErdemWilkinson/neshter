import type { ModelEntry } from '@/api/schemas';
import { t } from '@/i18n';
import { uiStore } from '@/core/stores/ui';

type Arch = NonNullable<ModelEntry['arch']>;

/** `24 katman · 14Q/2KV head · 4.864 MLP`: counts the header could not give are shown as `?`. */
export function formatArch(arch: Arch): string {
  const count = (n: number | null) => n ?? '?';
  return t('models.arch', {
    layers: count(arch.n_layers),
    q: count(arch.n_heads),
    kv: count(arch.n_kv_heads),
    mlp: count(arch.d_mlp),
  });
}

const UNITS = ['B', 'KB', 'MB', 'GB', 'TB'] as const;

/** `1,2 GB`: binary steps, one decimal from MB up, numbers in the current language. */
export function formatBytes(bytes: number): string {
  let value = Math.max(0, bytes);
  let unit = 0;
  while (value >= 1024 && unit < UNITS.length - 1) {
    value /= 1024;
    unit += 1;
  }
  const number = new Intl.NumberFormat(uiStore.getState().lang, { maximumFractionDigits: unit < 2 ? 0 : 1 });
  return `${number.format(value)} ${UNITS[unit]}`;
}

export function badgeLabel(badge: ModelEntry['badge']): string {
  switch (badge) {
    case 'ok':
      return t('models.badge.ok');
    case 'slow':
      return t('models.badge.slow');
    case 'no':
      return t('models.badge.no');
  }
}

/** The server's reason code for an unsupported model, in words (an unknown code reads as an unreadable architecture). */
export function reasonLabel(reason: string | null): string | null {
  if (reason === null) return null;
  return reason === 'unsupported_family' ? t('models.reason.unsupported_family') : t('models.reason.unknown_arch');
}
