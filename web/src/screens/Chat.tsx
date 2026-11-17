import { useEffect, useRef, useState } from 'react';
import { useStore } from 'zustand';
import { getApi } from '@/api/currentApi';
import { asPhoneApi, type ChatTurn } from '@/api/device/deviceApi';
import { ApiError, type ErrorCode } from '@/api/errors';
import { navigate } from '@/app/router';
import { sendableOps } from '@/core/basket';
import { basketStore } from '@/core/stores/basket';
import { sessionStore } from '@/core/stores/session';
import { useT } from '@/i18n';
import { Button } from '@/ui/Button';
import { ErrorCard } from '@/ui/ErrorCard';

type Turn = ChatTurn & { cut?: boolean; stopped?: boolean };
type Failure = { code: ErrorCode; detail: unknown };

/**
 * The phone's chat: a conversation with the open model, run on the phone. The basket's head cuts can be switched on
 * to talk to the cut model; each answer remembers whether it was written with them.
 */
export function Chat() {
  const t = useT();
  const model = useStore(sessionStore, (s) => s.model);
  const items = useStore(basketStore, (s) => s.items);
  const phone = asPhoneApi(getApi());
  const heads = sendableOps(items).flatMap((op) => (op.type === 'head' ? [[Number(op['layer']), Number(op['head'])] as [number, number]] : []));
  const headsOk = phone?.activeModel()?.heads_supported ?? false;

  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState('');
  const [live, setLive] = useState<string | null>(null);
  const [useCut, setUseCut] = useState(false);
  const [creative, setCreative] = useState(false);
  const [failure, setFailure] = useState<Failure | null>(null);
  const stopped = useRef(false);
  const end = useRef<HTMLDivElement>(null);
  const running = live !== null;
  const cutOn = useCut && headsOk && heads.length > 0;

  useEffect(() => {
    end.current?.scrollIntoView?.({ block: 'end' });
  }, [turns, live]);

  if (model === null || phone === null) {
    return (
      <section className="chat-screen">
        <h1>{t('chat.title')}</h1>
        <p>{t('chat.noModel')}</p>
        <Button variant="primary" onClick={() => navigate('models')}>
          {t('chat.toModels')}
        </Button>
      </section>
    );
  }

  const send = async () => {
    const text = draft.trim();
    if (text === '' || running) return;
    const history: Turn[] = [...turns, { role: 'user', content: text }];
    setTurns(history);
    setDraft('');
    setFailure(null);
    setLive('');
    stopped.current = false;
    let written = '';
    try {
      const answer = await phone.chat(
        {
          messages: history.map(({ role, content }) => ({ role, content })),
          cut: cutOn ? heads : [],
          temperature: creative ? 0.7 : 0,
        },
        (piece) => {
          written += piece;
          setLive(written);
        },
      );
      setTurns([...history, { role: 'assistant', content: answer, cut: cutOn }]);
    } catch (error) {
      if (stopped.current || (error instanceof ApiError && error.code === 'cancelled')) {
        setTurns([...history, { role: 'assistant', content: written, cut: cutOn, stopped: true }]);
      } else {
        setFailure(error instanceof ApiError ? { code: error.code, detail: error.detail } : { code: 'internal', detail: String(error) });
      }
    } finally {
      setLive(null);
    }
  };

  const stop = () => {
    stopped.current = true;
    void phone.stop();
  };

  return (
    <section className="chat-screen">
      <h1>{t('chat.title')}</h1>
      <p className="chat-lead">{t('chat.lead', { model: model.name })}</p>

      <div className="chat-options">
        {headsOk ? (
          heads.length > 0 ? (
            <label className="chat-option">
              <input type="checkbox" checked={useCut} disabled={running} onChange={(e) => setUseCut(e.target.checked)} />
              {t('chat.cut', { n: heads.length })}
            </label>
          ) : (
            <p className="chat-hint">{t('chat.cut.none')}</p>
          )
        ) : (
          <p className="chat-hint">{t('chat.cut.unsupported')}</p>
        )}
        <fieldset className="chat-style" disabled={running}>
          <legend>{t('chat.style')}</legend>
          <label>
            <input type="radio" name="chat-style" checked={!creative} onChange={() => setCreative(false)} />
            {t('chat.style.exact')}
          </label>
          <label>
            <input type="radio" name="chat-style" checked={creative} onChange={() => setCreative(true)} />
            {t('chat.style.creative')}
          </label>
        </fieldset>
      </div>

      <div className="chat-log glass" aria-live="polite">
        {turns.length === 0 && !running && <p className="chat-empty">{t('chat.empty')}</p>}
        {turns.map((turn, i) => (
          <div key={i} className={`chat-turn chat-${turn.role}`}>
            <span className="chat-who">
              {turn.role === 'user' ? t('chat.you') : t('chat.model')}
              {turn.cut === true && <span className="badge chat-cut-badge">{t('chat.cutBadge')}</span>}
            </span>
            <p className="chat-text">
              {turn.content}
              {turn.stopped === true && <span className="chat-stopped"> {t('chat.stopped')}</span>}
            </p>
          </div>
        ))}
        {running && (
          <div className="chat-turn chat-assistant is-live">
            <span className="chat-who">
              {t('chat.model')}
              {cutOn && <span className="badge chat-cut-badge">{t('chat.cutBadge')}</span>}
            </span>
            <p className="chat-text">
              {live}
              <span className="chat-caret" aria-hidden="true" />
            </p>
          </div>
        )}
        <div ref={end} />
      </div>

      {failure !== null && <ErrorCard code={failure.code} detail={failure.detail} />}

      <form
        className="chat-form"
        onSubmit={(e) => {
          e.preventDefault();
          void send();
        }}
      >
        <label className="sr-only" htmlFor="chat-input">
          {t('chat.input')}
        </label>
        <textarea
          id="chat-input"
          rows={2}
          value={draft}
          placeholder={t('chat.input')}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              void send();
            }
          }}
        />
        <div className="chat-actions">
          {running ? (
            <Button onClick={stop}>{t('chat.stop')}</Button>
          ) : (
            <Button type="submit" variant="primary" disabled={draft.trim() === ''}>
              {t('chat.send')}
            </Button>
          )}
          <Button disabled={running || turns.length === 0} onClick={() => setTurns([])}>
            {t('chat.new')}
          </Button>
        </div>
      </form>
    </section>
  );
}
