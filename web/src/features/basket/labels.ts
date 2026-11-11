import type { BasketItem } from '@/core/basket';
import { isDirectionOp, isHeadOp, isOpaque } from '@/core/ops';
import { t } from '@/i18n';

/**
 * The row text of a basket item. A direction op carries only its width (`d_model`), not the layer it was found at,
 * so it is labelled by size; an opaque op shows its `type` as plain text.
 */
export function itemLabel(item: BasketItem): string {
  const { op } = item;
  if (isHeadOp(op)) return t('basket.label.head', { layer: op.layer, head: op.head });
  if (isDirectionOp(op)) return t('basket.label.direction', { d: op.d_model });
  return isOpaque(op) ? t('basket.label.opaque', { type: op.type }) : t('recipe.item.other', { type: op.type });
}
