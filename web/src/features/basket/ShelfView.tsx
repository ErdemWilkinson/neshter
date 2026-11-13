import { useStore } from 'zustand';
import { basketStore } from '@/core/stores/basket';
import { isUnverifiedDirection } from '@/features/recipe/unverified';
import { useT } from '@/i18n';
import { Button } from '@/ui/Button';
import { announce } from '@/ui/LiveRegion';
import { itemLabel } from './labels';

/** `Keşif rafı`: explore-only cuts the save flow took out of the basket; they can come back from here. */
export function ShelfView() {
  const t = useT();
  const shelf = useStore(basketStore, (s) => s.shelf);
  if (shelf.length === 0) return null;
  return (
    <section className="shelf glass" aria-labelledby="shelf-title">
      <h2 id="shelf-title">{t('basket.shelf.title')}</h2>
      <ul className="basket-list">
        {shelf.map((item) => {
          const label = itemLabel(item);
          return (
            <li key={item.id} className="basket-row">
              <span className="basket-label">{label}</span>
              {isUnverifiedDirection(item) && (
                <span className="badge badge-unverified">{t('vocab.badge.unverified_direction')}</span>
              )}
              <Button
                aria-label={t('basket.shelf.restoreItem', { label })}
                onClick={() => {
                  if (!basketStore.getState().restoreFromShelf(item.id)) announce(t('basket.shelf.full'));
                }}
              >
                {t('basket.shelf.restore')}
              </Button>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
