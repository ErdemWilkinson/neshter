import { useT } from '@/i18n';
import { Button } from '@/ui/Button';
import { Dialog } from '@/ui/Dialog';

/** Only explore cuts are left: nothing can be saved, and no export is started. */
export function NothingToSaveDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const t = useT();
  return (
    <Dialog
      open={open}
      title={t('saveflow.nothing.title')}
      onCancel={onClose}
      actions={
        <Button variant="primary" onClick={onClose}>
          {t('ui.ok')}
        </Button>
      }
    >
      <p>{t('saveflow.nothing')}</p>
    </Dialog>
  );
}
