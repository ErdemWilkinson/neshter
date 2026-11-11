import { useDraftHeld } from '@/features/draft/useDraftHeld';
import { useT } from '@/i18n';
import { Button } from '@/ui/Button';
import { Dialog } from '@/ui/Dialog';

/** Unsaved cuts and the person is about to leave: save, leave (the server draft stays), or stay. */
export function LeaveDialog({
  open,
  onSave,
  onLeave,
  onCancel,
}: {
  open: boolean;
  onSave: () => void;
  onLeave: () => void;
  onCancel: () => void;
}) {
  const t = useT();
  const held = useDraftHeld();
  return (
    <Dialog
      open={open}
      title={t('leave.title')}
      onCancel={onCancel}
      actions={
        <>
          <Button variant="primary" onClick={onSave}>
            {t('leave.save')}
          </Button>
          <Button onClick={onLeave}>{t('leave.leave')}</Button>
          <Button variant="ghost" onClick={onCancel}>
            {t('leave.cancel')}
          </Button>
        </>
      }
    >
      <p>{t(held ? 'leave.body.held' : 'leave.body')}</p>
    </Dialog>
  );
}
