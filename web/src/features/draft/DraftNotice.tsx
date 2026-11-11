import { useStore } from 'zustand';
import { useT } from '@/i18n';
import { draftStatusStore } from './draftStatus';

/** Non-blocking note while autosave is refused as too large (413). */
export function DraftNotice() {
  const t = useT();
  const tooLarge = useStore(draftStatusStore, (s) => s.tooLarge);
  if (!tooLarge) return null;
  return (
    <div className="notice-banner" role="status">
      {t('draft.tooLarge')}
    </div>
  );
}
