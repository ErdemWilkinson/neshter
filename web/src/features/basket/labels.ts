import type { BasketItem } from '@/core/basket';
import { isDirectionOp, isHeadOp, isOpaque } from '@/core/ops';
import { t } from '@/i18n';

/** A hidden-state index as text: 0 is the embedding (`gömme`), index i is the output of block i - 1 (`K{i-1} sonrası`). */
export const layerPoint = (index: number): string =>
  index <= 0 ? t('concept.layer.embed') : t('concept.layer.after', { n: index - 1 });

/**
 * The row text of a basket item. A direction op carries only its width (`d_model`), not the layer it was found at,
 * so it is labelled by size; an opaque op shows its `type` as plain text.
 */
export function itemLabel(item: BasketItem): string {
  const { op } = item;
  if (isHeadOp(op)) return t('basket.label.head', { layer: op.layer, head: op.head });
  if (isDirectionOp(op)) {
    return item.foundAt === undefined
      ? t('basket.label.direction', { d: op.d_model })
      : t('basket.label.direction_at', { where: layerPoint(item.foundAt) });
  }
  return isOpaque(op) ? t('basket.label.opaque', { type: op.type }) : t('recipe.item.other', { type: op.type });
}
