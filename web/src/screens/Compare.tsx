import { useEffect, useMemo, useState } from 'react';
import { useStore } from 'zustand';
import type { JsonOp } from '@/api/client';
import { getApi } from '@/api/currentApi';
import type { Concept } from '@/api/schemas';
import { sendableOps } from '@/core/basket';
import { isSendable } from '@/core/ops';
import { basketConcept } from '@/core/basketConcept';
import { opsFingerprint } from '@/core/fingerprint';
import { basketStore } from '@/core/stores/basket';
import { compareStore } from '@/core/stores/compare';
import { isLocked, jobsStore } from '@/core/stores/jobs';
import { PairView } from '@/features/compare/PairView';
import { RefreshedBanner } from '@/features/compare/RefreshedBanner';
import { StatCards } from '@/features/compare/StatCards';
import { SuiteRunner } from '@/features/compare/SuiteRunner';
import { SuiteTable } from '@/features/compare/SuiteTable';
import { SummaryBanner } from '@/features/compare/SummaryBanner';
import { SaveFlowDialogs } from '@/features/saveflow/SaveFlowDialogs';
import { itemLabel } from '@/features/basket/labels';
import { useT } from '@/i18n';

/** Spec §2 step 4: the original and the cut model side by side, then the test set with honest uncertainty. */
export function Compare() {
  const t = useT();
  const items = useStore(basketStore, (s) => s.items);
  const lastSuite = useStore(compareStore, (s) => s.lastSuite);
  const locked = useStore(jobsStore, isLocked);
  const [concepts, setConcepts] = useState<Concept[]>([]);

  useEffect(() => {
    let alive = true;
    getApi()
      .concepts()
      .then((list) => alive && setConcepts(list))
      .catch(() => undefined); // without concepts there is simply no concept id or suite target
    return () => {
      alive = false;
    };
  }, []);

  const ops = useMemo(() => sendableOps(items), [items]);
  const { conceptId, suiteTarget } = useMemo(
    () =>
      basketConcept(
        items.filter((i) => isSendable(i.op) && i.pendingDiscard !== true),
        concepts,
      ),
    [items, concepts],
  );
  const cuts = useMemo(() => items.filter((i) => isSendable(i.op) && i.pendingDiscard !== true), [items]);
  const empty = ops.length === 0;
  const stale = lastSuite !== null && lastSuite.fp !== opsFingerprint(ops);
  const summary = lastSuite?.result.summary ?? null;
  // A summary exists only when the run had a target category; without one nothing is marked as the target.
  const target = summary === null ? null : suiteTarget;

  return (
    <section className="compare-screen">
      <h1>{t('app.step.compare')}</h1>
      <RefreshedBanner />
      {empty && <p>{t('compare.none')}</p>}
      {!empty && (
        <div className="compare-cuts">
          <span id="compare-cuts-label">{t('compare.cuts')}:</span>
          <ul className="chip-list" aria-labelledby="compare-cuts-label">
            {cuts.map((item) => (
              <li key={item.id} className="chip num">
                {itemLabel(item)}
              </li>
            ))}
          </ul>
        </div>
      )}
      <PairView ops={ops as JsonOp[]} conceptId={conceptId} disabled={empty || locked} />
      <SuiteRunner ops={ops} target={suiteTarget} disabled={empty || locked} />
      {stale && (
        <p className="stale-banner" role="status">
          {t('compare.stale')}
        </p>
      )}
      {lastSuite !== null && (
        <div className={stale ? 'suite-result suite-result-stale' : 'suite-result'}>
          <SummaryBanner summary={lastSuite.result.summary} />
          <StatCards result={lastSuite.result} target={target} />
          <SuiteTable result={lastSuite.result} target={target} />
        </div>
      )}
      <SaveFlowDialogs />
    </section>
  );
}
