import type { BasketItem } from '@/core/basket';
import { isOpaque, isSavable } from '@/core/ops';
import { isUnverifiedDirection } from '@/features/recipe/unverified';
import { useT } from '@/i18n';
import { Button } from '@/ui/Button';
import { itemLabel } from './labels';

function Kind({ item }: { item: BasketItem }) {
  const t = useT();
  if (isSavable(item.op)) return <span className="badge badge-savable">{t('basket.badge.savable')}</span>;
  if (isOpaque(item.op)) return <span className="badge badge-unsupported">{t('basket.badge.unsupported')}</span>;
  return <span className="badge badge-explore">{t('basket.badge.explore')}</span>;
}

/** One basket item: its label, what it can be used for, and (while the save flow drops it) `atılacak`. */
export function BasketRow({ item, onRemove }: { item: BasketItem; onRemove: (id: string) => void }) {
  const t = useT();
  const label = itemLabel(item);
  return (
    <li className={`basket-row${item.pendingDiscard ? ' basket-row-pending' : ''}`}>
      <span className="basket-label">{label}</span>
      <Kind item={item} />
      {isUnverifiedDirection(item) && (
        <span className="badge badge-unverified">{t('vocab.badge.unverified_direction')}</span>
      )}
      {item.pendingDiscard && <span className="badge badge-pending">{t('basket.pending')}</span>}
      <Button variant="ghost" aria-label={t('basket.remove', { label })} onClick={() => onRemove(item.id)}>
        ×
      </Button>
    </li>
  );
}
