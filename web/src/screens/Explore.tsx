import { useEffect, useId, useMemo, useRef, useState } from 'react';
import { useStore } from 'zustand';
import { getApi } from '@/api/currentApi';
import { getExplorePrompt, setExplorePrompt } from '@/app/sessionStorage';
import { ApiError, BusyError, type ErrorCode } from '@/api/errors';
import type { NodeId } from '@/core/ids';
import { headId } from '@/core/ids';
import { isLocked, jobsStore } from '@/core/stores/jobs';
import { basketStore } from '@/core/stores/basket';
import { selectionStore } from '@/core/stores/selection';
import { sessionStore } from '@/core/stores/session';
import { uiStore } from '@/core/stores/ui';
import type { BasketItem } from '@/core/basket';
import { isHeadOp } from '@/core/ops';
import { normalizeGlobal, normalizePerLayer, sliceToken } from '@/core/normalize';
import { BasketPanel } from '@/features/basket/BasketPanel';
import { ShelfView } from '@/features/basket/ShelfView';
import { SelectionPanel } from '@/features/measure/SelectionPanel';
import { DEMO_HEADS, DEMO_PROMPT, demoStore } from '@/features/demo/demoStore';
import { exploreStore, type Unit } from '@/features/map/exploreStore';
import { Heatmap } from '@/features/map/Heatmap';
import { HeatmapTable } from '@/features/map/HeatmapTable';
import { Hotspots, topHeads } from '@/features/map/Hotspots';
import { MlpList } from '@/features/map/MlpList';
import { ScaleToggle } from '@/features/map/ScaleToggle';
import { TokenStrip } from '@/features/map/TokenStrip';
import { ViewSwitch, effectiveView } from '@/features/map/ViewSwitch';
import { GraphPanel } from '@/features/graph/GraphPanel';
import { isDeviceApp } from '@/api/device/bridge';
import { AttentionPanel } from '@/features/insight/AttentionPanel';
import { LensPanel, shownToken } from '@/features/insight/LensPanel';
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

  // After a reload the text field gets its prompt back (tab-only convenience; the heatmap itself is not kept).
  useEffect(() => {
    if (exploreStore.getState().prompt !== '') return;
    const saved = getExplorePrompt();
    if (saved !== null) exploreStore.getState().setPrompt(saved);
  }, []);

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
  // Hot spots rank heads across the whole map, so they always use the global scale.
  const hotValues = useMemo(() => {
    if (result === null || chosen < 0) return null;
    try {
      return normalizeGlobal(sliceToken(result.powerRaw, chosen));
    } catch {
      return null;
    }
  }, [result, chosen]);
  const dims = useMemo(
    () => ({ n_layers: values?.length ?? 0, n_heads: values?.[0]?.length ?? 0 }),
    [values],
  );

  const run = async (text: string = prompt) => {
    const id = ++runId.current;
    setRunning(true);
    setFailure(null);
    try {
      const response = await getApi().activity({ prompt: text });
      if (id !== runId.current) return;
      exploreStore.getState().setResult({ prompt: text, tokens: response.tokens, powerRaw: response.power_raw });
    } catch (error) {
      if (id !== runId.current) return;
      if (demoStore.getState().step === 'pick') demoStore.getState().setStep('idle'); // the example stops here
      // Busy is the banner's business (the client flags it too; this also covers a client without the hook).
      if (error instanceof BusyError) sessionStore.getState().setBusy(true);
      else setFailure(asFailure(error));
    } finally {
      if (id === runId.current) setRunning(false);
    }
  };

  // The example from Home: send its text once, then pick the strongest heads at the last token for the measure step.
  const demoStep = useStore(demoStore, (s) => s.step);
  const demoShown = useStore(demoStore, (s) => s.shown);
  const model = useStore(sessionStore, (s) => s.model);
  useEffect(() => {
    if (demoStore.getState().step !== 'run' || model === null) return; // the live step: StrictMode runs this twice
    demoStore.getState().setStep('pick');
    exploreStore.getState().setPrompt(DEMO_PROMPT);
    setExplorePrompt(DEMO_PROMPT);
    queueMicrotask(() => void run(DEMO_PROMPT)); // after the effect: the run sets this component's state
    // eslint-disable-next-line react-hooks/exhaustive-deps -- one run per example; `run` is a fresh closure each render
  }, [demoStep, model]);
  useEffect(() => {
    if (demoStep !== 'pick' || hotValues === null || result?.prompt !== DEMO_PROMPT) return;
    selectionStore.getState().set(topHeads(hotValues, DEMO_HEADS).map((row) => row.id));
    demoStore.getState().setStep('measure');
  }, [demoStep, hotValues, result]);

  const unitOptions: { value: Unit; label: string }[] = [
    { value: 'head', label: t('map.unit.head') },
    ...(isDeviceApp() ? [] : [{ value: 'mlp' as const, label: t('map.unit.mlp') }]),
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
          onChange={(event) => {
            demoStore.getState().stop(); // their own text: the example's explanation no longer fits
            exploreStore.getState().setPrompt(event.target.value);
            setExplorePrompt(event.target.value);
          }}
        />
        <Button type="submit" variant="primary" disabled={locked || running || prompt.trim() === ''}>
          {t('map.run')}
        </Button>
      </form>

      {demoShown && <p className="demo-note glass">{t('demo.note', { n: DEMO_HEADS })}</p>}
      {running && <p role="status">{t('map.running')}</p>}
      {failure !== null && <ErrorCard code={failure.code} detail={failure.detail} />}

      {result === null && !running && failure === null && <p className="explore-empty">{t('map.empty')}</p>}

      <div className={`explore-desk${view === '3d' ? ' is-wide' : ''}`}>
      <div className="explore-main">
      {result !== null && (
        <>
          <div className="explore-controls">
            <ViewSwitch />
            {(view === '2d' || view === 'list') && <ScaleToggle />}
            {unitOptions.length > 1 && ( // the phone has heads only: no choice to offer
              <div className="control">
                <span aria-hidden="true">{t('map.unit')}</span>
                <Segmented
                  label={t('map.unit')}
                  options={unitOptions}
                  value={unit}
                  onChange={(next) => exploreStore.getState().setUnit(next)}
                />
              </div>
            )}
          </div>
          <TokenStrip
            tokens={tokens}
            index={chosen}
            disabled={locked}
            onSelect={(i) => exploreStore.getState().setTokenIndex(i)}
          />
          {view === '3d' ? (
            <GraphPanel prompt={result.prompt} heat={hotValues} basket={basket} />
          ) : unit === 'mlp' ? (
            chosen >= 0 && <MlpList prompt={result.prompt} tokenIndex={chosen} />
          ) : values === null || dims.n_layers === 0 ? (
            <ErrorCard code="internal" />
          ) : view === 'hot' ? (
            hotValues !== null && (
              <Hotspots values={hotValues} basket={basket} token={shownToken(tokens[chosen] ?? '')} />
            )
          ) : (
            <>
              <p id={legendId} className="map-legend">
                <span>{scale === 'layer' ? t('map.legend') : t('map.legend.global')}</span>{' '}
                <span className="badge-estimated">{t('map.badge.estimated')}</span>
                <span className="legend-scale" aria-hidden="true">
                  <span>{t('map.legend.low')}</span>
                  <span className="legend-bar" />
                  <span>{t('map.legend.high')}</span>
                </span>
                <span className="map-axes">{t('map.axes')}</span>
              </p>
              {view === 'list' ? (
                <HeatmapTable values={values} basket={basket} />
              ) : (
                <Heatmap dims={dims} values={values} basket={basket} describedBy={legendId} />
              )}
            </>
          )}
          {chosen >= 0 && (
            <div className="insight-row">
              <LensPanel prompt={result.prompt} position={chosen} disabled={locked} />
              <AttentionPanel
                prompt={result.prompt}
                position={chosen}
                nLayers={dims.n_layers}
                nHeads={dims.n_heads}
                disabled={locked}
              />
            </div>
          )}
        </>
      )}
      </div>
      <aside className="explore-side" aria-label={t('explore.side')}>
        <SelectionPanel values={values} />
        <BasketPanel />
      </aside>
      </div>
      <ShelfView />
    </section>
  );
}
