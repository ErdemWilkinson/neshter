import { lazy, Suspense, useEffect, useId, useMemo, useRef, useState } from 'react';
import { useStore } from 'zustand';
import { getApi } from '@/api/currentApi';
import { ApiError, BusyError, isAbortError, type ErrorCode } from '@/api/errors';
import { graphResult, type GraphCut, type GraphEdge, type GraphResult } from '@/api/schemas';
import { useJob } from '@/api/useJob';
import { isMotionReduced, systemMatchMedia } from '@/app/prefs';
import type { NodeId } from '@/core/ids';
import { isLocked, jobsStore } from '@/core/stores/jobs';
import { selectionStore } from '@/core/stores/selection';
import { sessionStore } from '@/core/stores/session';
import { uiStore } from '@/core/stores/ui';
import { formatFixed } from '@/features/map/format';
import { shownToken } from '@/features/insight/LensPanel';
import { useT, type MessageKey } from '@/i18n';
import { Button } from '@/ui/Button';
import { ErrorCard } from '@/ui/ErrorCard';
import { ProgressCard } from '@/ui/ProgressCard';
import type { Hover, Pick } from './drawFlow';
import type { SceneMode } from './FlowScene';
import { incomingShare, liveWeights, placeNodes, trunkWidths, visibleEdges } from './layout';

interface Failure {
  code: ErrorCode;
  detail: unknown;
}
const asFailure = (error: unknown): Failure =>
  error instanceof ApiError ? { code: error.code, detail: error.detail } : { code: 'internal', detail: String(error) };

const CHANNEL: Record<GraphEdge['channel'], MessageKey> = {
  q: 'graph.channel.q',
  v: 'graph.channel.v',
  resid: 'graph.channel.resid',
  logit: 'graph.channel.logit',
};

// Loaded on first use so the heatmap screens do not carry the scene code.
const FlowScene = lazy(() => import('./FlowScene'));

const DEFAULT_LIMIT = 40;
const LABELLED_HEADS = 8;

function useNodeName() {
  const t = useT();
  return (graph: GraphResult | null, id: string): string => {
    const [tag, a, b] = id.split(':');
    if (tag === 'h') return `K${a}·H${b}`;
    if (tag === 'n') return t('graph.node.neuron', { layer: a ?? '', index: b ?? '' });
    if (tag === 't') return shownToken(graph?.tokens[Number(a)]?.text ?? '');
    return shownToken(graph?.metric.text ?? '');
  };
}

/**
 * Plan 3 M2 "3B bağlar". Opens on the heatmap seen from above; drawing the estimated attribution graph of one
 * answer tilts it into the connection graph, each head on its heatmap cell. Rivers can be cut for real (explore
 * only); heads go to the selection, and from there to the basket, exactly as on the heatmap.
 */
export function GraphPanel({
  prompt,
  heat,
  basket,
}: {
  prompt: string;
  /** The heatmap's normalised values [layer][head] for the chosen token, or null. */
  heat: number[][] | null;
  /** Heads that already have a cut in the basket. */
  basket: ReadonlySet<NodeId>;
}) {
  const t = useT();
  const name = useNodeName();
  const sliderId = useId();
  const pickId = useId();
  const { job, start, cancel, reset } = useJob<GraphResult>();
  const locked = useStore(jobsStore, isLocked);
  const chosen = useStore(selectionStore, (s) => s.selected);
  const reducedPref = useStore(uiStore, (s) => s.reducedMotion);
  const reducedMotion = isMotionReduced(reducedPref, systemMatchMedia());
  const [graph, setGraph] = useState<GraphResult | null>(null);
  const [mode, setMode] = useState<SceneMode>('heat');
  const [failure, setFailure] = useState<Failure | null>(null);
  const [limit, setLimit] = useState(DEFAULT_LIMIT);
  const [selected, setSelected] = useState<Pick | null>(null);
  const [hover, setHover] = useState<Hover | null>(null);
  const [cut, setCut] = useState<string[]>([]);
  const [report, setReport] = useState<GraphCut | null>(null);
  const [cutFailure, setCutFailure] = useState<Failure | null>(null);
  const [measuring, setMeasuring] = useState(false);
  const alive = useRef(true);
  useEffect(() => () => void (alive.current = false), []);

  const draw = async () => {
    setFailure(null);
    setGraph(null);
    setMode('heat');
    setCut([]);
    setReport(null);
    setSelected(null);
    try {
      const final = await start(() => getApi().graph({ prompt }));
      if (!alive.current) return;
      if (final.state === 'cancelled') return reset();
      if (final.state !== 'done') return setFailure({ code: final.error_code ?? 'internal', detail: final.error });
      const parsed = graphResult.safeParse(final.result);
      if (!parsed.success) return setFailure({ code: 'internal', detail: parsed.error.issues });
      setGraph(parsed.data);
      setMode('flow');
    } catch (error) {
      if (isAbortError(error) || !alive.current) return;
      if (error instanceof BusyError) sessionStore.getState().setBusy(true);
      else setFailure(asFailure(error));
    }
  };

  const grid = useMemo(
    () => ({ nLayers: graph?.n_layers ?? heat?.length ?? 0, nHeads: graph?.n_heads ?? heat?.[0]?.length ?? 0 }),
    [graph, heat],
  );
  const nodes = useMemo(() => (graph === null ? [] : placeNodes(graph)), [graph]);
  const cutSet = useMemo(() => new Set(cut), [cut]);
  const edges = useMemo(() => (graph === null ? [] : visibleEdges(graph.edges, limit, cutSet)), [graph, limit, cutSet]);
  const edgeById = useMemo(() => new Map((graph?.edges ?? []).map((e) => [e.id, e])), [graph]);
  const live = useMemo(() => (graph === null ? new Map<string, number>() : liveWeights(graph.nodes, graph.edges, cutSet)), [graph, cutSet]);
  const maxWeight = useMemo(() => Math.max(1e-9, ...(graph?.edges ?? []).map((e) => Math.abs(e.weight))), [graph]);
  const trunk = useMemo(() => trunkWidths(graph?.rivers.trunk ?? []), [graph]);
  const heads = useMemo(
    () => nodes.filter((n) => n.kind === 'head').sort((a, b) => Math.abs(b.score) - Math.abs(a.score)),
    [nodes],
  );
  const labels = useMemo(
    () => new Set<string>([...heads.slice(0, LABELLED_HEADS).map((n) => n.id), ...chosen]),
    [heads, chosen],
  );

  const measure = async (ids: string[]) => {
    if (graph === null) return;
    setCutFailure(null);
    if (ids.length === 0) return setReport(null);
    setMeasuring(true);
    try {
      const r = await getApi().graphCut(graph.graph_id, ids);
      if (alive.current) setReport(r);
    } catch (error) {
      if (!alive.current) return;
      if (error instanceof BusyError) sessionStore.getState().setBusy(true);
      else setCutFailure(asFailure(error));
    } finally {
      if (alive.current) setMeasuring(false);
    }
  };

  const cutEdges = (ids: string[]) => {
    const fresh = ids.filter((id) => !cutSet.has(id));
    if (fresh.length === 0) return;
    const next = [...cut, ...fresh].slice(0, 64);
    setCut(next);
    void measure(next);
  };
  const restore = (id: string) => {
    const next = cut.filter((x) => x !== id);
    setCut(next);
    void measure(next);
  };

  /** A heatmap cell picks the head into the selection, as on the heatmap; in the graph a click selects. */
  const onSelect = (pick: Pick | null) => {
    if (pick?.kind === 'cell') {
      selectionStore.getState().toggle(pick.id as NodeId);
      setSelected(nodes.some((n) => n.id === pick.id) ? { kind: 'node', id: pick.id } : null);
      return;
    }
    setSelected(pick);
  };

  const running = job !== null && (job.state === 'queued' || job.state === 'running');
  const showStage = graph !== null || (heat !== null && grid.nLayers > 0);
  const selEdge = selected?.kind === 'edge' ? edgeById.get(selected.id) ?? null : null;
  const selNode = selected?.kind === 'node' ? nodes.find((n) => n.id === selected.id) ?? null : null;
  const outgoing = selNode === null ? [] : edges.filter((e) => e.src === selNode.id && !cutSet.has(e.id));
  const hoverText = (() => {
    if (hover === null) return null;
    const { pick } = hover;
    if (pick.kind !== 'edge') {
      const [, l, h] = pick.id.split(':');
      const power = pick.kind === 'cell' ? heat?.[Number(l)]?.[Number(h)] : undefined;
      return power === undefined ? name(graph, pick.id) : `${name(graph, pick.id)} · ${formatFixed(power, 2)}`;
    }
    const e = edgeById.get(pick.id);
    return e === undefined ? null : `${name(graph, e.src)} → ${name(graph, e.dst)} · e = ${formatFixed(e.weight, 3)}`;
  })();
  const pickValue = selected === null || selected.kind === 'cell' ? '' : `${selected.kind}:${selected.id}`;

  return (
    <section className="graph-panel" aria-labelledby="graph-title">
      <div className="insight-head">
        <h2 id="graph-title">{t('graph.title')}</h2>
        <Button variant="primary" disabled={locked || running || prompt.trim() === ''} onClick={() => void draw()}>
          {graph === null ? t('graph.draw') : t('graph.redraw')}
        </Button>
      </div>
      <p className="field-hint">{t('graph.lead')}</p>
      {running && job !== null && <ProgressCard job={job} onCancel={() => void cancel()} />}
      {failure !== null && <ErrorCard code={failure.code} detail={failure.detail} />}

      {graph !== null && (
        <>
          <p className="graph-meta">
            {t('graph.metric', { answer: shownToken(graph.metric.text) })}{' '}
            <span className="badge-estimated">{t('map.badge.estimated')}</span>{' '}
            <span className="field-hint">
              {t('graph.pruning', { nodes: graph.pruning.kept_nodes, edges: graph.pruning.kept_edges })} ·{' '}
              {t('graph.noK')}
            </span>
          </p>
          <div className="graph-tools">
            <label htmlFor={sliderId}>{t('graph.limit', { n: Math.min(limit, graph.edges.length) })}</label>
            <input
              id={sliderId}
              type="range"
              min={5}
              max={Math.max(5, graph.edges.length)}
              step={5}
              value={Math.min(limit, Math.max(5, graph.edges.length))}
              onChange={(e) => setLimit(Number(e.target.value))}
            />
          </div>
        </>
      )}

      {showStage && (
        <div className="graph-layout">
          <div className="graph-stage" role="img" aria-label={t('graph.stageLabel', { edges: edges.length })}>
            <Suspense fallback={null}>
            <FlowScene
              grid={grid}
              nodes={nodes}
              edges={edges}
              live={live}
              maxWeight={maxWeight}
              trunk={trunk}
              heat={heat}
              mode={mode}
              cut={cutSet}
              basket={basket}
              chosen={chosen}
              selected={selected}
              labels={labels}
              reducedMotion={reducedMotion}
              name={(id) => name(graph, id)}
              onSelect={onSelect}
              onHover={setHover}
            />
            </Suspense>
            <div className="graph-mode" role="group" aria-label={t('graph.mode.label')}>
              <button type="button" aria-pressed={mode === 'heat'} onClick={() => setMode('heat')}>
                {t('graph.mode.heat')}
              </button>
              <button type="button" aria-pressed={mode === 'flow'} disabled={graph === null} onClick={() => setMode('flow')}>
                {t('graph.mode.flow')}
              </button>
            </div>
            {hoverText !== null && hover !== null && (
              <div className="graph-tip" style={{ left: hover.x, top: hover.y }} aria-hidden="true">
                {hoverText}
              </div>
            )}
            <p className="graph-legend" aria-hidden="true">
              {mode === 'heat' ? (
                <>
                  {t('map.legend.low')} <span className="sw sw-heat" /> {t('map.legend.high')} · {t('graph.legend.heat')}
                </>
              ) : (
                <>
                  <span className="sw sw-inc" /> {t('graph.legend.inc')} <span className="sw sw-dec" /> {t('graph.legend.dec')}{' '}
                  <span className="sw sw-cut" /> {t('graph.legend.cut')} <span className="sw sw-basket" /> {t('graph.legend.basket')} ·{' '}
                  {t('graph.legend.trunk')}
                </>
              )}
            </p>
          </div>

          <div className="graph-side">
            {graph === null ? (
              <section className="graph-card">
                <p className="field-hint">{t('graph.heatFirst')}</p>
              </section>
            ) : (
              <>
                <section className="graph-card" aria-live="polite">
                  <label htmlFor={pickId} className="field-hint">
                    {t('graph.pickList')}
                  </label>
                  <select
                    id={pickId}
                    value={pickValue}
                    onChange={(event) => {
                      const v = event.target.value;
                      const at = v.indexOf(':');
                      const kind = v.slice(0, at);
                      setSelected(kind === 'node' || kind === 'edge' ? { kind, id: v.slice(at + 1) } : null);
                    }}
                  >
                    <option value="">{t('graph.pickList.none')}</option>
                    <optgroup label={t('graph.pickList.heads')}>
                      {heads.slice(0, 40).map((n) => (
                        <option key={n.id} value={`node:${n.id}`}>
                          {name(graph, n.id)}
                        </option>
                      ))}
                    </optgroup>
                    <optgroup label={t('graph.pickList.edges')}>
                      {edges.map((e) => (
                        <option key={e.id} value={`edge:${e.id}`}>
                          {`${name(graph, e.src)} → ${name(graph, e.dst)} (${formatFixed(e.weight, 2)})`}
                        </option>
                      ))}
                    </optgroup>
                  </select>
                  {selEdge !== null ? (
                    <>
                      <h3>{t('graph.edge.title')}</h3>
                      <dl className="graph-kv">
                        <dt>{t('graph.edge.from')}</dt>
                        <dd className="num">{name(graph, selEdge.src)}</dd>
                        <dt>{t('graph.edge.to')}</dt>
                        <dd className="num">{name(graph, selEdge.dst)}</dd>
                        <dt>{t('graph.edge.channel')}</dt>
                        <dd>{t(CHANNEL[selEdge.channel])}</dd>
                        <dt>{t('graph.edge.weight')}</dt>
                        <dd>
                          {t(selEdge.weight >= 0 ? 'graph.edge.raises' : 'graph.edge.lowers', { answer: shownToken(graph.metric.text) })}{' '}
                          <span className="num">e = {formatFixed(selEdge.weight, 3)}</span>
                        </dd>
                      </dl>
                      <p className="graph-preview">
                        {t('graph.edge.share', {
                          pct: Math.round(incomingShare(graph.edges, selEdge) * 100),
                          to: name(graph, selEdge.dst),
                        })}
                      </p>
                      <Button
                        className="btn-cut"
                        disabled={cutSet.has(selEdge.id) || locked || measuring}
                        onClick={() => cutEdges([selEdge.id])}
                      >
                        {cutSet.has(selEdge.id) ? t('graph.edge.isCut') : t('graph.edge.cut')}
                      </Button>
                      <p className="field-hint">{t('graph.edge.exploreOnly')}</p>
                    </>
                  ) : selNode !== null ? (
                    <>
                      <h3>{name(graph, selNode.id)}</h3>
                      <p>
                        {t('graph.node.score')} <span className="num">{formatFixed(selNode.score, 3)}</span>
                      </p>
                      {selNode.kind === 'head' && (
                        <>
                          {basket.has(selNode.id as NodeId) ? (
                            <p className="field-hint">{t('graph.head.inBasket')}</p>
                          ) : (
                            <Button onClick={() => selectionStore.getState().toggle(selNode.id as NodeId)}>
                              {chosen.has(selNode.id as NodeId) ? t('graph.head.unselect') : t('graph.head.select')}
                            </Button>
                          )}
                          <p className="field-hint">{t('graph.head.selectHint')}</p>
                        </>
                      )}
                      {selNode.kind !== 'output' && (
                        <>
                          <p className="field-hint">{t('graph.node.cutAllHint', { n: outgoing.length })}</p>
                          <Button
                            className="btn-cut"
                            disabled={outgoing.length === 0 || locked || measuring}
                            onClick={() => cutEdges(outgoing.map((e) => e.id))}
                          >
                            {t('graph.node.cutAll', { n: outgoing.length })}
                          </Button>
                        </>
                      )}
                    </>
                  ) : (
                    <p className="field-hint">{t('graph.pickHint')}</p>
                  )}
                </section>

                <section className="graph-card">
                  <h3>{t('graph.cuts.title', { n: cut.length })}</h3>
                  {cut.length === 0 ? (
                    <p className="field-hint">{t('graph.cuts.empty')}</p>
                  ) : (
                    <ul className="basket-list">
                      {cut.map((id) => {
                        const e = edgeById.get(id);
                        if (e === undefined) return null;
                        const label = `${name(graph, e.src)} → ${name(graph, e.dst)}`;
                        return (
                          <li key={id} className="basket-row">
                            <span className="basket-label num">{label}</span>
                            <span className="badge badge-explore">{t('basket.badge.explore')}</span>
                            <Button variant="ghost" aria-label={t('graph.cuts.undo', { label })} onClick={() => restore(id)}>
                              ×
                            </Button>
                          </li>
                        );
                      })}
                    </ul>
                  )}
                  {measuring && <p role="status">{t('graph.cuts.measuring')}</p>}
                  {cutFailure !== null && <ErrorCard code={cutFailure.code} detail={cutFailure.detail} />}
                  {report !== null && cut.length > 0 && !measuring && (
                    <div className="graph-report">
                      <p>
                        <b>
                          {t('graph.report.p', {
                            answer: shownToken(graph.metric.text),
                            before: Math.round(report.p_before * 100),
                            after: Math.round(report.p_after * 100),
                          })}
                        </b>
                      </p>
                      <p className="num">
                        {t('graph.report.delta', {
                          real: formatFixed(report.delta_metric, 3),
                          est: formatFixed(report.delta_estimated, 3),
                          kl: formatFixed(report.kl, 4),
                        })}
                      </p>
                      <p>
                        {t('graph.report.top')}{' '}
                        {report.top_after.slice(0, 3).map((tok, i) => (
                          <span key={tok.token_id}>
                            {i > 0 && ', '}
                            <code>{shownToken(tok.text)}</code> <span className="num">%{Math.round(tok.p * 100)}</span>
                          </span>
                        ))}
                      </p>
                      <p className="field-hint">{t('graph.report.note')}</p>
                    </div>
                  )}
                </section>
              </>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
