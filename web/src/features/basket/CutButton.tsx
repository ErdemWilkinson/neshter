import { useId } from 'react';
import { useStore } from 'zustand';
import { newBasketItem } from '@/core/basket';
import { opsFingerprint } from '@/core/fingerprint';
import type { HeadOp } from '@/core/ops';
import { basketStore } from '@/core/stores/basket';
import { measureStore } from '@/core/stores/measure';
import { useT } from '@/i18n';
import { Button } from '@/ui/Button';
import { announce } from '@/ui/LiveRegion';

/**
 * "Kes (geçici)": puts the selected heads into the basket as zero-ablation cuts. Only after the real effect of
 * exactly this selection was measured (spec §2 step 3a: silence only after measuring). The file is never changed.
 */
export function CutButton({ selection }: { selection: readonly HeadOp[] }) {
  const t = useT();
  const reasonId = useId();
  const fp = opsFingerprint(selection);
  const measurement = useStore(measureStore, (s) => (selection.length === 0 ? undefined : s.get(fp)));
  const enabled = measurement !== undefined;

  const cut = () => {
    if (measurement === undefined) return;
    const now = new Date().toISOString();
    const added = basketStore
      .getState()
      .add(selection.map((op) => ({ ...newBasketItem(op, { origin: 'heatmap', concept: null, addedAt: now }), measurement })));
    announce(added > 0 ? t('cut.added', { n: added }) : t('cut.alreadyIn'));
  };

  return (
    <div className="cut">
      <Button
        variant="primary"
        className="cut-button"
        disabled={!enabled}
        aria-describedby={enabled ? undefined : reasonId}
        onClick={cut}
      >
        <span>{t('cut.button')}</span>
        <small className="cut-sub">{t('cut.sub')}</small>
      </Button>
      {!enabled && (
        <p id={reasonId} className="cut-reason">
          {t('cut.needMeasure')}
        </p>
      )}
    </div>
  );
}
