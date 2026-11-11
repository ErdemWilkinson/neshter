import { useStore } from 'zustand';
import { isOpaque } from '@/core/ops';
import type { BasketItem } from '@/core/basket';
import { basketStore } from '@/core/stores/basket';
import { useT } from '@/i18n';
import { Button } from '@/ui/Button';
import { Dialog } from '@/ui/Dialog';

/** Opaque ops (written by a newer version) have no label this version could know. Text only, never markup. */
export function exploreLabel(item: BasketItem, opaqueLabel: string): string {
  return isOpaque(item.op) ? opaqueLabel : item.op.type;
}

/** "These N cuts are for exploring only..." for the explore items in `exploreIds`; one question, no second modal. */
export function DiscardDialog({
  open,
  exploreIds,
  onConfirm,
  onCancel,
}: {
  open: boolean;
  exploreIds: readonly string[];
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const t = useT();
  const items = useStore(basketStore, (s) => s.items);
  const labels = items.filter((item) => exploreIds.includes(item.id)).map((item) => exploreLabel(item, t('saveflow.opaque')));
  return (
    <Dialog
      open={open}
      title={t('saveflow.discard.title')}
      onCancel={onCancel}
      actions={
        <>
          <Button variant="primary" onClick={onConfirm}>
            {t('saveflow.discard.confirm')}
          </Button>
          <Button onClick={onCancel}>{t('saveflow.discard.cancel')}</Button>
        </>
      }
    >
      <p>{t('saveflow.discard.body', { n: exploreIds.length, list: labels.join(', ') })}</p>
    </Dialog>
  );
}
