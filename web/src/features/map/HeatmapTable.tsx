import { useMemo, useState } from 'react';
import { useStore } from 'zustand';
import { headId, type NodeId } from '@/core/ids';
import { selectionStore } from '@/core/stores/selection';
import { useT } from '@/i18n';
import { formatPower } from './format';

type Sort = 'none' | 'desc' | 'asc';
const NEXT_SORT: Record<Sort, Sort> = { none: 'desc', desc: 'asc', asc: 'none' };
const ARIA_SORT = { none: 'none', desc: 'descending', asc: 'ascending' } as const;

interface Row {
  layer: number;
  head: number;
  id: NodeId;
  power: number;
}

/** The Liste view: one row per head, the same selection store as the heatmap. */
export function HeatmapTable({ values, basket }: { values: number[][]; basket: ReadonlySet<NodeId> }) {
  const t = useT();
  const selected = useStore(selectionStore, (s) => s.selected);
  const [sort, setSort] = useState<Sort>('none');

  const rows = useMemo(() => {
    const all: Row[] = values.flatMap((row, layer) =>
      row.map((power, head) => ({ layer, head, id: headId(layer, head), power })),
    );
    if (sort === 'none') return all;
    const sign = sort === 'desc' ? -1 : 1;
    return [...all].sort((a, b) => sign * (a.power - b.power) || a.layer - b.layer || a.head - b.head);
  }, [values, sort]);

  return (
    <table className="heatmap-table">
      <caption className="sr-only">{t('map.table.caption')}</caption>
      <thead>
        <tr>
          <th scope="col">{t('map.table.layer')}</th>
          <th scope="col">{t('map.table.head')}</th>
          <th scope="col" aria-sort={ARIA_SORT[sort]}>
            <button type="button" className="sort-button" onClick={() => setSort(NEXT_SORT[sort])}>
              {t('map.table.power')}
              <span aria-hidden="true">{sort === 'desc' ? ' ▼' : sort === 'asc' ? ' ▲' : ''}</span>
            </button>
          </th>
          <th scope="col">{t('map.table.selected')}</th>
          <th scope="col">{t('map.table.basket')}</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.id} aria-selected={selected.has(row.id)}>
            <td className="num">{row.layer}</td>
            <td className="num">{row.head}</td>
            <td className="num">{formatPower(row.power)}</td>
            <td>
              <input
                type="checkbox"
                checked={selected.has(row.id)}
                aria-label={t('map.table.select', { layer: row.layer, head: row.head })}
                onChange={() => selectionStore.getState().toggle(row.id)}
              />
            </td>
            <td>{basket.has(row.id) ? t('map.table.inBasket') : ''}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
