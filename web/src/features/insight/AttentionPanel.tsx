import { useId, useState, type CSSProperties } from 'react';
import { useStore } from 'zustand';
import { getApi } from '@/api/currentApi';
import type { AttentionResponse } from '@/api/schemas';
import { parseNodeId } from '@/core/ids';
import { selectionStore } from '@/core/stores/selection';
import { useT } from '@/i18n';
import { Button } from '@/ui/Button';
import { ErrorCard } from '@/ui/ErrorCard';
import { shownToken } from './LensPanel';
import { useInsight } from './useInsight';

const MAX_GRID = 48; // longer prompts show only the strongest sources of the chosen token

/** The head to start from: the focused heatmap cell, else the first selected head, else K0·H0. */
function startHead(focus: string | null, selected: ReadonlySet<string>): { layer: number; head: number } {
  for (const id of [focus, ...selected]) {
    const parsed = id === null ? null : parseNodeId(id);
    if (parsed?.kind === 'head') return { layer: parsed.layer, head: parsed.head };
  }
  return { layer: 0, head: 0 };
}

/** One head's attention pattern: which earlier tokens each token reads (rows: reading token, columns: source). */
export function AttentionPanel({
  prompt,
  position,
  nLayers,
  nHeads,
  disabled,
}: {
  prompt: string;
  position: number;
  nLayers: number;
  nHeads: number;
  disabled: boolean;
}) {
  const t = useT();
  const titleId = useId();
  const layerId = useId();
  const headSelectId = useId();
  const focus = useStore(selectionStore, (s) => s.focus);
  const selected = useStore(selectionStore, (s) => s.selected);
  const [open, setOpen] = useState(false);
  const [pick, setPick] = useState<{ layer: number; head: number } | null>(null);
  const { layer, head } = pick ?? startHead(focus, selected);
  const state = useInsight<AttentionResponse>(open, `${prompt}\u0000${layer}`, () => getApi().attention({ prompt, layer }));
  const data = state.status === 'ready' ? state.data : null;
  const pattern = data?.patterns[head] ?? null;
  const row = pattern?.[position] ?? null;
  const strongest = row === null ? [] : row.map((p, k) => ({ k, p })).sort((a, b) => b.p - a.p).slice(0, 3);
  const tok = (k: number) => shownToken(data?.tokens[k] ?? '');

  return (
    <section className="insight-panel" aria-labelledby={titleId}>
      <div className="insight-head">
        <h2 id={titleId}>{t('attn.title')}</h2>
        <Button aria-expanded={open} disabled={disabled && !open} onClick={() => setOpen((o) => !o)}>
          {open ? t('insight.hide') : t('insight.show')}
        </Button>
      </div>
      <p className="field-hint">{t('attn.lead')}</p>
      {open && (
        <div className="attn-pick">
          <label htmlFor={layerId}>{t('map.axis.layer')}</label>
          <select id={layerId} value={layer} onChange={(e) => setPick({ layer: Number(e.target.value), head })}>
            {Array.from({ length: nLayers }, (_, i) => (
              <option key={i} value={i}>
                {i}
              </option>
            ))}
          </select>
          <label htmlFor={headSelectId}>{t('map.axis.head')}</label>
          <select id={headSelectId} value={head} onChange={(e) => setPick({ layer, head: Number(e.target.value) })}>
            {Array.from({ length: nHeads }, (_, i) => (
              <option key={i} value={i}>
                {i}
              </option>
            ))}
          </select>
        </div>
      )}
      {open && state.status === 'loading' && <p role="status">{t('map.running')}</p>}
      {state.status === 'failed' && <ErrorCard code={state.code} detail={state.detail} />}
      {open && data !== null && pattern !== null && (
        <>
          <p>
            {t('attn.summary', { layer, head, token: tok(position) })}{' '}
            {strongest.map((s, i) => (
              <span key={s.k}>
                {i > 0 && ', '}
                <code>{tok(s.k)}</code> <span className="num">%{Math.round(s.p * 100)}</span>
              </span>
            ))}
          </p>
          {data.tokens.length <= MAX_GRID ? (
            // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex -- scrollable region must be focusable
            <div className="table-scroll" role="region" tabIndex={0} aria-label={t('attn.gridLabel', { layer, head })}>
              <table className="attn-grid">
                <caption className="sr-only">{t('attn.gridLabel', { layer, head })}</caption>
                <thead>
                  <tr>
                    <td />
                    {data.tokens.map((_, k) => (
                      <th key={k} scope="col">
                        <span className="attn-col">{tok(k)}</span>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {pattern.map((cells, q) => (
                    <tr key={q} className={q === position ? 'attn-row-chosen' : undefined}>
                      <th scope="row">{tok(q)}</th>
                      {/* a token reads only itself and earlier tokens: the matrix is a triangle, later cells stay empty */}
                      {cells.map((p, k) =>
                        k > q ? (
                          <td key={k} className="attn-none" />
                        ) : (
                          <td
                            key={k}
                            title={`${tok(q)} → ${tok(k)}: %${Math.round(p * 100)}`}
                            style={{ '--w': Math.sqrt(Math.max(0, p)) } as CSSProperties}
                          />
                        ),
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="field-hint">{t('attn.tooLong', { n: MAX_GRID })}</p>
          )}
        </>
      )}
    </section>
  );
}
