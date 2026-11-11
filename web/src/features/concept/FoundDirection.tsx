import { useState } from 'react';
import type { HuntResult } from '@/api/schemas';
import { newBasketItem } from '@/core/basket';
import { DirectionOpSchema } from '@/core/ops';
import { basketStore } from '@/core/stores/basket';
import { layerPoint } from '@/features/basket/labels';
import { useT } from '@/i18n';
import { Button } from '@/ui/Button';
import { ErrorCard } from '@/ui/ErrorCard';

/**
 * The direction a `direction` hunt found. It enters the basket as an explore item (`Sepete ekle (geçici)`): the
 * concept it came from is kept (`refusal` for a safety concept), and the layer it was found at is shown on the row.
 */
export function FoundDirection({ result, conceptId }: { result: HuntResult; conceptId: string }) {
  const t = useT();
  const [outcome, setOutcome] = useState<'added' | 'present' | null>(null);
  const { direction, direction_shape: shape } = result;
  if (direction === null || shape === null || shape[0] === undefined) return null;

  const op = DirectionOpSchema.safeParse({ type: 'direction', d_model: shape[0], vector_b64: direction });
  if (!op.success) return <ErrorCard code="internal" detail={op.error.issues} />;

  const layer = result.direction_layer ?? result.ref_layer;
  const add = () => {
    const item = newBasketItem(op.data, { origin: 'concept', concept: conceptId, addedAt: new Date().toISOString() });
    setOutcome(basketStore.getState().add([{ ...item, foundAt: layer }]) > 0 ? 'added' : 'present');
  };

  return (
    <section className="found-direction" aria-labelledby="found-direction-title">
      <h3 id="found-direction-title">{t('concept.direction.title')}</h3>
      <p>{t('concept.direction.layer', { where: layerPoint(layer) })}</p>
      <Button variant="primary" onClick={add}>
        {t('concept.direction.add')}
      </Button>
      {outcome !== null && (
        <p role="status">{outcome === 'added' ? t('concept.direction.added') : t('concept.direction.notAdded')}</p>
      )}
    </section>
  );
}
