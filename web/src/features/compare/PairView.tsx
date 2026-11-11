import { useEffect, useRef, useState } from 'react';
import { useStore } from 'zustand';
import type { JsonOp } from '@/api/client';
import { getApi } from '@/api/currentApi';
import { ApiError, BusyError, type ErrorCode } from '@/api/errors';
import { compareStore } from '@/core/stores/compare';
import { sessionStore } from '@/core/stores/session';
import { useT } from '@/i18n';
import { Button } from '@/ui/Button';
import { ErrorCard } from '@/ui/ErrorCard';

const MAX_NEW_TOKENS = 200;

function Side({ title, text, hit }: { title: string; text: string; hit: boolean | null }) {
  const t = useT();
  return (
    <section className="pair-side glass">
      <h3>{title}</h3>
      {hit !== null && (
        <span className={`badge ${hit ? 'badge-hit' : 'badge-miss'}`}>{t(hit ? 'compare.pair.hit' : 'compare.pair.miss')}</span>
      )}
      {/* Model output is attacker-influenced: always React text, never markup. */}
      <p className="pair-text">{text}</p>
    </section>
  );
}

/**
 * The same prompt answered by the original and the cut model, side by side (temperature 0, seed 42).
 * Only sendable ops are sent; `disabled` covers an empty basket and a job holding the model.
 */
export function PairView({ ops, conceptId, disabled }: { ops: JsonOp[]; conceptId: string | null; disabled: boolean }) {
  const t = useT();
  const pair = useStore(compareStore, (s) => s.lastPair);
  const [prompt, setPrompt] = useState(() => compareStore.getState().lastPrompt ?? '');
  const [asking, setAsking] = useState(false);
  const [failure, setFailure] = useState<{ code: ErrorCode; detail: unknown } | null>(null);
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  const ask = async () => {
    setAsking(true);
    setFailure(null);
    try {
      const answer = await getApi().compare({
        prompt,
        recipe_ops: ops,
        concept_id: conceptId,
        max_new_tokens: MAX_NEW_TOKENS,
      });
      compareStore.getState().setLastPrompt(prompt);
      compareStore.getState().setLastPair(answer);
    } catch (error) {
      if (!alive.current) return;
      // Busy is the banner's business: the model is held by a job, not broken.
      if (error instanceof BusyError) sessionStore.getState().setBusy(true);
      else setFailure(error instanceof ApiError ? { code: error.code, detail: error.detail } : { code: 'internal', detail: String(error) });
    } finally {
      if (alive.current) setAsking(false);
    }
  };

  return (
    <section className="pair-view">
      <h2>{t('compare.pair.title')}</h2>
      <label className="pair-prompt">
        <span>{t('compare.pair.prompt')}</span>
        <textarea rows={3} value={prompt} onChange={(event) => setPrompt(event.target.value)} />
      </label>
      <Button variant="primary" disabled={disabled || asking || prompt.trim() === ''} onClick={() => void ask()}>
        {asking ? t('compare.pair.asking') : t('compare.pair.ask')}
      </Button>
      {failure !== null && (
        <>
          <p>{t('compare.pair.failed')}</p>
          <ErrorCard code={failure.code} detail={failure.detail} />
        </>
      )}
      {pair !== null && (
        <>
          <div className="pair-columns">
            <Side title={t('compare.pair.original')} text={pair.original} hit={pair.target_hit?.original ?? null} />
            <Side title={t('compare.pair.cut')} text={pair.modified} hit={pair.target_hit?.modified ?? null} />
          </div>
          <p className="pair-mode num">{t('compare.pair.mode', { mode: pair.mode })}</p>
        </>
      )}
    </section>
  );
}
