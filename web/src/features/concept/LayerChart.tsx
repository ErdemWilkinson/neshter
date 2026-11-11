import type { HuntResult } from '@/api/schemas';
import { layerPoint } from '@/features/basket/labels';
import { useT } from '@/i18n';

const clamp01 = (x: number): number => (Number.isFinite(x) ? Math.min(1, Math.max(0, x)) : 0);

/**
 * One bar per hidden state (`n_layers + 1`): index 0 is the embedding, index i the output of block i - 1.
 * The reference layer's bar is marked.
 */
export function LayerChart({ scores, refLayer }: { scores: readonly number[]; refLayer: number }) {
  const t = useT();
  return (
    <section className="layer-chart" aria-labelledby="layer-chart-title">
      <h3 id="layer-chart-title">{t('concept.layers.title')}</h3>
      <ol aria-label={t('concept.layers.list')} className="layer-chart-bars">
        {scores.map((score, index) => (
          <li key={index} className={index === refLayer ? 'layer-chart-row layer-chart-ref' : 'layer-chart-row'}>
            <span className="layer-chart-label">{layerPoint(index)}</span>
            <span className="layer-chart-track" aria-hidden="true">
              <span className="layer-chart-bar" style={{ width: `${clamp01(score) * 100}%` }} />
            </span>
            <span className="num layer-chart-value">{score.toFixed(2)}</span>
            {index === refLayer && <span className="badge">{t('concept.layer.ref')}</span>}
          </li>
        ))}
      </ol>
    </section>
  );
}

const percent = (x: number): number => Math.round(clamp01(x) * 100);

/** How well the found signal separates the held-out 20 %. */
export function ValidationLine({ result }: { result: Pick<HuntResult, 'val_accuracy' | 'val_ci' | 'val_n'> }) {
  const t = useT();
  return (
    <p className="num validation-line">
      {t('concept.validation', {
        a: percent(result.val_accuracy),
        lo: percent(result.val_ci[0]),
        hi: percent(result.val_ci[1]),
        n: result.val_n,
      })}
    </p>
  );
}
