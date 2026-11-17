import { useId, useState } from 'react';
import { getApi } from '@/api/currentApi';
import type { LensResponse } from '@/api/schemas';
import { layerPoint } from '@/features/basket/labels';
import { useT } from '@/i18n';
import { Button } from '@/ui/Button';
import { ErrorCard } from '@/ui/ErrorCard';
import { useInsight } from './useInsight';

const pct = (p: number): string => `${Math.round(p * 100)}`;
/** A token as one line of text: blank -> ␣, line breaks -> ↵ (a raw newline would stretch the row), tabs -> ⇥. */
export const shownToken = (text: string): string =>
  text.trim() === '' ? '␣' : text.replace(/\r?\n/g, '↵').replace(/\t/g, '⇥');

/**
 * Logit lens: after every layer the residual stream is read out through the final norm and the unembedding, so
 * the person sees what the model "would answer" at that depth and where the final answer appears.
 */
export function LensPanel({ prompt, position, disabled }: { prompt: string; position: number; disabled: boolean }) {
  const t = useT();
  const titleId = useId();
  const [open, setOpen] = useState(false);
  const state = useInsight<LensResponse>(open, `${prompt}\u0000${position}`, () => getApi().lens({ prompt, position }));
  const data = state.status === 'ready' ? state.data : null;
  const firstHit = data === null ? -1 : data.layers.findIndex((row) => row.rank_final === 0);
  // the rival: the strongest guess after the last layer that is not the answer (orange next to the answer's green)
  const rival = data === null ? undefined : data.layers.at(-1)?.top.find((tok) => tok.token_id !== data.final.token_id);
  const kind = (id: number, j: number): string =>
    data !== null && id === data.final.token_id ? 'is-final' : id === rival?.token_id ? 'is-rival' : j === 0 ? 'is-top' : '';

  return (
    <section className="insight-panel" aria-labelledby={titleId}>
      <div className="insight-head">
        <h2 id={titleId}>{t('lens.title')}</h2>
        <Button aria-expanded={open} disabled={disabled && !open} onClick={() => setOpen((o) => !o)}>
          {open ? t('insight.hide') : t('insight.show')}
        </Button>
      </div>
      <p className="field-hint">{t('lens.lead')}</p>
      {open && state.status === 'loading' && <p role="status">{t('map.running')}</p>}
      {state.status === 'failed' && <ErrorCard code={state.code} detail={state.detail} />}
      {open && data !== null && (
        <>
          <p>
            {t('lens.final', { answer: shownToken(data.final.text), token: shownToken(data.tokens[data.position] ?? '') })}{' '}
            {firstHit >= 0 && t('lens.firstHit', { where: layerPoint(firstHit) })}
          </p>
          {/* eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex -- scrollable region must be focusable */}
          <div className="table-scroll" role="region" tabIndex={0} aria-label={t('lens.title')}>
            <table className="lens-table">
              <thead>
                <tr>
                  <th scope="col">{t('lens.col.layer')}</th>
                  <th scope="col">{t('lens.col.top')}</th>
                  <th scope="col">{t('lens.col.final', { answer: shownToken(data.final.text) })}</th>
                </tr>
              </thead>
              <tbody>
                {data.layers.map((row, i) => (
                  <tr key={row.layer} className={i === firstHit ? 'lens-hit' : undefined}>
                    <th scope="row">{layerPoint(i)}</th>
                    <td className="lens-top">
                      {/* the ladder: the three guesses side by side, each as wide as its share; the final answer green, its rival orange */}
                      <span className="lens-stack" aria-hidden="true">
                        {row.top.slice(0, 3).map((tok, j) => (
                          <i
                            key={tok.token_id}
                            className={kind(tok.token_id, j) || undefined}
                            style={{ width: `${Math.max(0, tok.p * 100)}%` }}
                          />
                        ))}
                      </span>
                      {row.top.slice(0, 3).map((tok, j) => (
                        <span key={tok.token_id} className={`lens-token ${kind(tok.token_id, j)}`.trim()}>
                          <code>{shownToken(tok.text)}</code> <span className="num">%{pct(tok.p)}</span>
                        </span>
                      ))}
                    </td>
                    <td className="lens-final">
                      <span className="lens-bar" aria-hidden="true">
                        <i style={{ width: `${Math.max(1, row.p_final * 100)}%` }} />
                      </span>
                      <span className="num">%{pct(row.p_final)}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <ul className="lens-key">
            <li>
              <i className="is-final" aria-hidden="true" /> {t('lens.key.final', { answer: shownToken(data.final.text) })}
            </li>
            {rival !== undefined && (
              <li>
                <i className="is-rival" aria-hidden="true" /> {t('lens.key.rival', { token: shownToken(rival.text) })}
              </li>
            )}
            <li>
              <i aria-hidden="true" /> {t('lens.key.other')}
            </li>
          </ul>
        </>
      )}
    </section>
  );
}
