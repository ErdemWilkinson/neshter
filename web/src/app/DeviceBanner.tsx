import { useStore } from 'zustand';
import { sessionStore } from '@/core/stores/session';
import { useT } from '@/i18n';

/** Shown while the loaded model carries a device note (GPU did not fit, slow bf16 on this CPU). */
export function DeviceBanner() {
  const t = useT();
  const note = useStore(sessionStore, (s) => s.model?.device_note ?? null);
  if (note === null || note === '') return null;
  return (
    <aside className="device-banner" aria-label={t('app.device.title')}>
      <strong>{t('app.device.title')}</strong> {note}
    </aside>
  );
}
