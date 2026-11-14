import { useMemo } from 'react';
import { useStore } from 'zustand';
import { headId, type NodeId } from '@/core/ids';
import { selectionStore } from '@/core/stores/selection';
import { useT } from '@/i18n';
import { Button } from '@/ui/Button';
import { formatPower } from './format';

export const HOTSPOT_COUNT = 10;

/**
 * The strongest heads at the chosen token, strongest first, each with a bar and a one-click pick: the shortest way
 * from "which heads matter here?" to a selection. `values` must be scaled over the whole map (not per layer), or
 * every layer's top head would tie at 1.
 */
export function Hotspots({
  values,
  basket,
  token,
}: {
  values: number[][];
  basket: ReadonlySet<NodeId>;
  token: string;
}) {
  const t = useT();
  const selected = useStore(selectionStore, (s) => s.selected);
  const top = useMemo(
    () =>
      values
        .flatMap((row, layer) => row.map((power, head) => ({ layer, head, power, id: headId(layer, head) })))
        .sort((a, b) => b.power - a.power || a.layer - b.layer || a.head - b.head)
        .slice(0, HOTSPOT_COUNT),
    [values],
  );

  return (
    <div className="hotspots">
      <ol className="hotspot-list" aria-label={t('map.hot.label', { n: top.length, token })}>
        {top.map((row, i) => {
          const picked = selected.has(row.id);
          return (
            <li key={row.id} className={`hotspot${picked ? ' is-picked' : ''}`}>
              <span className="hotspot-rank num" aria-hidden="true">
                {i + 1}
              </span>
              <span className="hotspot-id num">{`K${row.layer}·H${row.head}`}</span>
              <span className="hotspot-bar" aria-hidden="true">
                <i style={{ width: `${Math.max(2, row.power * 100)}%` }} />
              </span>
              <span className="hotspot-power num">{formatPower(row.power)}</span>
              {basket.has(row.id) ? (
                <span className="badge badge-basket">{t('map.table.inBasket')}</span>
              ) : (
                <Button
                  aria-pressed={picked}
                  aria-label={t('map.table.select', { layer: row.layer, head: row.head })}
                  onClick={() => selectionStore.getState().toggle(row.id)}
                >
                  {picked ? t('map.hot.picked') : t('map.hot.pick')}
                </Button>
              )}
            </li>
          );
        })}
      </ol>
      <p className="field-hint">{t('map.hot.hint')}</p>
    </div>
  );
}
