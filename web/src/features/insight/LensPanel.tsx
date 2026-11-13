import { useId, useState } from 'react';
import { getApi } from '@/api/currentApi';
import type { LensResponse } from '@/api/schemas';
import { layerPoint } from '@/features/basket/labels';
import { useT } from '@/i18n';
import { Button } from '@/ui/Button';
import { ErrorCard } from '@/ui/ErrorCard';
import { useInsight } from './useInsight';

const pct = (p: number): string => `${Math.round(p * 100)}`;
export const shownToken = (text: string): string => (text.trim() === '' ? '␣' : text);

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
                      {row.top.slice(0, 3).map((tok) => (
                        <span key={tok.token_id} className="lens-token">
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
        </>
      )}
    </section>
  );
}
