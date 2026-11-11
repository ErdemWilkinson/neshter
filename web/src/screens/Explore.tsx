import { useId, useMemo, useRef, useState } from 'react';
import { useStore } from 'zustand';
import { getApi } from '@/api/currentApi';
import { ApiError, BusyError, type ErrorCode } from '@/api/errors';
import type { NodeId } from '@/core/ids';
import { headId } from '@/core/ids';
import { isLocked, jobsStore } from '@/core/stores/jobs';
import { basketStore } from '@/core/stores/basket';
import { sessionStore } from '@/core/stores/session';
import { uiStore } from '@/core/stores/ui';
import type { BasketItem } from '@/core/basket';
import { isHeadOp } from '@/core/ops';
import { normalizeGlobal, normalizePerLayer, sliceToken } from '@/core/normalize';
import { BasketPanel } from '@/features/basket/BasketPanel';
import { ShelfView } from '@/features/basket/ShelfView';
import { SelectionPanel } from '@/features/measure/SelectionPanel';
import { exploreStore, type Unit } from '@/features/map/exploreStore';
import { Heatmap } from '@/features/map/Heatmap';
import { HeatmapTable } from '@/features/map/HeatmapTable';
import { MlpList } from '@/features/map/MlpList';
import { ScaleToggle } from '@/features/map/ScaleToggle';
import { TokenStrip } from '@/features/map/TokenStrip';
import { ViewSwitch, effectiveView } from '@/features/map/ViewSwitch';
import { useT } from '@/i18n';
import { Button } from '@/ui/Button';
import { ErrorCard } from '@/ui/ErrorCard';
import { Segmented } from '@/ui/Segmented';

interface Failure {
  code: ErrorCode;
  detail: unknown;
}

const asFailure = (error: unknown): Failure =>
  error instanceof ApiError ? { code: error.code, detail: error.detail } : { code: 'internal', detail: String(error) };

/** The heatmap cells that already have a head cut in the basket. */
function basketHeadIds(items: readonly BasketItem[]): ReadonlySet<NodeId> {
  const ids = new Set<NodeId>();
  for (const { op } of items) if (isHeadOp(op)) ids.add(headId(op.layer, op.head));
  return ids;
}

/** Spec §2 step 3a: type a text, see how strongly every head writes at one token, pick heads from the map or the list. */
export function Explore() {
  const t = useT();
  const legendId = useId();
  const promptId = useId();
  const prompt = useStore(exploreStore, (s) => s.prompt);
  const result = useStore(exploreStore, (s) => s.result);
  const tokenIndex = useStore(exploreStore, (s) => s.tokenIndex);
  const scale = useStore(exploreStore, (s) => s.scale);
  const unit = useStore(exploreStore, (s) => s.unit);
  const view = useStore(uiStore, (s) => effectiveView(s.view));
  const locked = useStore(jobsStore, isLocked);
  const items = useStore(basketStore, (s) => s.items);
  const basket = useMemo(() => basketHeadIds(items), [items]);
  const [running, setRunning] = useState(false);
  const [failure, setFailure] = useState<Failure | null>(null);
  const runId = useRef(0);

  const tokens = result?.tokens ?? [];
  const chosen = tokens.length === 0 ? -1 : Math.min(tokenIndex ?? tokens.length - 1, tokens.length - 1);

  // Re-slicing and re-normalising are client-side: `power_raw` holds every token already.
  const values = useMemo(() => {
    if (result === null || chosen < 0) return null;
    try {
      const slice = sliceToken(result.powerRaw, chosen);
      return scale === 'layer' ? normalizePerLayer(slice) : normalizeGlobal(slice);
    } catch {
      return null; // a ragged payload: shown as "unreadable" below, never a crash
    }
  }, [result, chosen, scale]);
  const dims = useMemo(
    () => ({ n_layers: values?.length ?? 0, n_heads: values?.[0]?.length ?? 0 }),
    [values],
  );

  const run = async () => {
    const id = ++runId.current;
    setRunning(true);
    setFailure(null);
    try {
      const text = prompt;
      const response = await getApi().activity({ prompt: text });
      if (id !== runId.current) return;
      exploreStore.getState().setResult({ prompt: text, tokens: response.tokens, powerRaw: response.power_raw });
    } catch (error) {
      if (id !== runId.current) return;
      // Busy is the banner's business (the client flags it too; this also covers a client without the hook).
      if (error instanceof BusyError) sessionStore.getState().setBusy(true);
      else setFailure(asFailure(error));
    } finally {
      if (id === runId.current) setRunning(false);
    }
  };

  const unitOptions: { value: Unit; label: string }[] = [
    { value: 'head', label: t('map.unit.head') },
    { value: 'mlp', label: t('map.unit.mlp') },
  ];

  return (
    <section className="explore-screen">
      <h1>{t('app.step.explore')}</h1>
      <form
        className="explore-prompt"
        onSubmit={(event) => {
          event.preventDefault();
          if (!locked && !running && prompt.trim() !== '') void run();
        }}
      >
        <label htmlFor={promptId}>{t('map.prompt')}</label>
        <textarea
          id={promptId}
          rows={2}
          value={prompt}
          onChange={(event) => exploreStore.getState().setPrompt(event.target.value)}
        />
        <Button type="submit" variant="primary" disabled={locked || running || prompt.trim() === ''}>
          {t('map.run')}
        </Button>
      </form>

      {running && <p role="status">{t('map.running')}</p>}
      {failure !== null && <ErrorCard code={failure.code} detail={failure.detail} />}

      {result === null && !running && failure === null && <p className="explore-empty">{t('map.empty')}</p>}

      {result !== null && (
        <>
          <div className="explore-controls">
            <ViewSwitch />
            <ScaleToggle />
            <div className="control">
              <span aria-hidden="true">{t('map.unit')}</span>
              <Segmented
                label={t('map.unit')}
                options={unitOptions}
                value={unit}
                onChange={(next) => exploreStore.getState().setUnit(next)}
              />
            </div>
          </div>
          <TokenStrip
            tokens={tokens}
            index={chosen}
            disabled={locked}
            onSelect={(i) => exploreStore.getState().setTokenIndex(i)}
          />
          {unit === 'mlp' ? (
            chosen >= 0 && <MlpList prompt={result.prompt} tokenIndex={chosen} />
          ) : values === null || dims.n_layers === 0 ? (
            <ErrorCard code="internal" />
          ) : (
            <>
              <p id={legendId} className="map-legend">
                <span>{scale === 'layer' ? t('map.legend') : t('map.legend.global')}</span>{' '}
                <span className="badge-estimated">{t('map.badge.estimated')}</span>
                <span className="map-axes">{t('map.axes')}</span>
              </p>
              {view === 'list' ? (
                <HeatmapTable values={values} basket={basket} />
              ) : (
                <Heatmap dims={dims} values={values} basket={basket} describedBy={legendId} />
              )}
            </>
          )}
        </>
      )}

      <SelectionPanel values={values} />
      <BasketPanel />
      <ShelfView />
    </section>
  );
}
