import { useMemo } from 'react';
import { useStore } from 'zustand';
import { headId, parseNodeId } from '@/core/ids';
import { opsFingerprint } from '@/core/fingerprint';
import { measureStore } from '@/core/stores/measure';
import { selectionStore } from '@/core/stores/selection';
import { CutButton } from '@/features/basket/CutButton';
import { formatPower } from '@/features/map/format';
import { useT } from '@/i18n';
import { Button } from '@/ui/Button';
import { TechDetail } from '@/ui/TechDetail';
import { ControlStatsCard } from './ControlStatsCard';
import { MeasureButton } from './MeasureButton';
import { selectedHeadOps } from './selectionOps';

/** The heads picked on the map or in the list, what zeroing them means, and the measure → cut steps. */
export function SelectionPanel({ values }: { values: number[][] | null }) {
  const t = useT();
  const selected = useStore(selectionStore, (s) => s.selected);
  const focus = useStore(selectionStore, (s) => s.focus);
  const ops = useMemo(() => selectedHeadOps(selected), [selected]);
  const fp = useMemo(() => opsFingerprint(ops), [ops]);
  const measurement = useStore(measureStore, (s) => (ops.length === 0 ? undefined : s.get(fp)));

  const node = focus === null ? null : parseNodeId(focus);
  const focusPower = node?.kind === 'head' ? values?.[node.layer]?.[node.head] : undefined;

  return (
    <section className="selection-panel glass" aria-labelledby="selection-title">
      <h2 id="selection-title">{t('selection.title')}</h2>
      {ops.length === 0 ? (
        <p>{t('selection.empty')}</p>
      ) : (
        <ul className="chip-list" aria-label={t('selection.list')}>
          {ops.map((op) => (
            <li key={headId(op.layer, op.head)} className="chip">
              <span className="num">{`K${op.layer}·H${op.head}`}</span>
              <Button
                variant="ghost"
                aria-label={t('selection.remove', { layer: op.layer, head: op.head })}
                onClick={() => selectionStore.getState().toggle(headId(op.layer, op.head))}
              >
                ×
              </Button>
            </li>
          ))}
        </ul>
      )}
      {node?.kind === 'head' && focusPower !== undefined && (
        <p className="num">
          {t('selection.focus', { layer: node.layer, head: node.head, value: formatPower(focusPower) })}
        </p>
      )}
      <p>{t('measure.zero')}</p>
      <TechDetail>{t('measure.zero.tech')}</TechDetail>
      <MeasureButton ops={ops} />
      {measurement !== undefined && <ControlStatsCard m={measurement} />}
      <CutButton selection={ops} />
    </section>
  );
}
