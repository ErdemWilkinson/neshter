import { useStore } from 'zustand';
import { isDeviceApp } from '@/api/device/bridge';
import { navigate } from '@/app/router';
import { countExploreOnly, countSavable, sendableOps, unsavedCount } from '@/core/basket';
import { isSavable } from '@/core/ops';
import { basketStore } from '@/core/stores/basket';
import { plural, useT } from '@/i18n';
import { Button } from '@/ui/Button';
import { BasketRow } from './BasketRow';

/**
 * The basket: every cut, what each can be used for, undo for the last removal, and the way on to Karşılaştır / Kaydet.
 * `Modeli kaydet` only navigates: the save flow starts on the Save screen (`SAVE_CLICKED` belongs to its `Kaydet`).
 */
export function BasketPanel() {
  const t = useT();
  const items = useStore(basketStore, (s) => s.items);
  const lastRemoved = useStore(basketStore, (s) => s.lastRemoved);
  const savable = countSavable(items);
  const exploreOnly = countExploreOnly(items);
  const savedOps = useStore(basketStore, (s) => s.savedOps);
  // Savable cuts that the last save already wrote out: they stay in the basket but are no longer "to do".
  const savedCount = savable - unsavedCount(items.filter((item) => isSavable(item.op)), savedOps);

  return (
    <section className="basket-panel glass" aria-labelledby="basket-title">
      <h2 id="basket-title">{t('basket.title')}</h2>
      {items.length === 0 ? (
        <p>{t('basket.empty')}</p>
      ) : (
        <ul className="basket-list">
          {items.map((item) => (
            <BasketRow key={item.id} item={item} onRemove={(id) => basketStore.getState().remove(id)} />
          ))}
        </ul>
      )}
      {lastRemoved !== null && (
        <Button onClick={() => basketStore.getState().undoRemove()}>{t('basket.undo')}</Button>
      )}
      <p className="num basket-line">{t('basket.line', { s: savable, e: exploreOnly })}
        {savedCount > 0 && ` · ${t('basket.line.saved', { n: savedCount })}`}
      </p>
      <div className="basket-actions">
        <Button disabled={sendableOps(items).length === 0} onClick={() => navigate('compare')}>
          {t('basket.compare')}
        </Button>
        {!isDeviceApp() && ( // the phone app does not write model files
          <Button variant="primary" disabled={savable === 0} onClick={() => navigate('save')}>
            {plural('basket.save', savable)}
          </Button>
        )}
      </div>
    </section>
  );
}
