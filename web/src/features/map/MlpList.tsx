import { useEffect, useState } from 'react';
import { getApi } from '@/api/currentApi';
import { ApiError, BusyError, isAbortError, type ErrorCode } from '@/api/errors';
import { sessionStore } from '@/core/stores/session';
import { useT } from '@/i18n';
import { ErrorCard } from '@/ui/ErrorCard';
import { formatFixed } from './format';

export const MLP_TOP = 50;

interface Neuron {
  layer: number;
  neuron: number;
  value: number;
}
type Outcome = { status: 'ready'; rows: Neuron[] } | { status: 'failed'; code: ErrorCode; detail: unknown };
/** The outcome belongs to one request; until the current request answers, the list is loading. */
type State = { key: string; outcome: Outcome } | null;

/** The strongest MLP neurons for one token of the prompt that produced the heatmap. */
export function MlpList({ prompt, tokenIndex }: { prompt: string; tokenIndex: number }) {
  const t = useT();
  const [state, setState] = useState<State>(null);
  const key = `${tokenIndex}:${prompt}`;

  useEffect(() => {
    let current = true;
    getApi()
      .mlp({ prompt, token_index: tokenIndex })
      .then((rows) => {
        if (current) setState({ key, outcome: { status: 'ready', rows: rows.slice(0, MLP_TOP) } });
      })
      .catch((error: unknown) => {
        if (!current || isAbortError(error)) return;
        // Busy is the banner's business (the client already flagged it), not an error card.
        if (error instanceof BusyError) {
          sessionStore.getState().setBusy(true);
          setState({ key, outcome: { status: 'ready', rows: [] } });
        } else if (error instanceof ApiError) {
          setState({ key, outcome: { status: 'failed', code: error.code, detail: error.detail } });
        } else {
          setState({ key, outcome: { status: 'failed', code: 'internal', detail: String(error) } });
        }
      });
    return () => {
      current = false;
    };
  }, [key, prompt, tokenIndex]);

  if (state === null || state.key !== key) return <p>{t('map.mlp.loading')}</p>;
  const { outcome } = state;
  if (outcome.status === 'failed') return <ErrorCard code={outcome.code} detail={outcome.detail} />;
  return (
    <ol className="mlp-list" aria-label={t('map.mlp.label')}>
      {outcome.rows.map((row) => (
        <li key={`${row.layer}:${row.neuron}`} className="num">
          {t('map.mlp.row', { layer: row.layer, neuron: row.neuron, value: formatFixed(row.value, 3) })}
        </li>
      ))}
    </ol>
  );
}
