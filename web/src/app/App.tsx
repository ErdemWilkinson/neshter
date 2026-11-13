import { useEffect, useState } from 'react';
import { useStore } from 'zustand';
import { createApi } from '@/api/client';
import { getApi, setApi } from '@/api/currentApi';
import { ApiError, type ErrorCode } from '@/api/errors';
import { startHeartbeat } from '@/api/heartbeat';
import { basketStore } from '@/core/stores/basket';
import { compareStore } from '@/core/stores/compare';
import { sessionStore } from '@/core/stores/session';
import { startAutosave } from '@/features/draft/autosave';
import { draftStatusStore } from '@/features/draft/draftStatus';
import { useLeaveGuard } from '@/features/leave/useLeaveGuard';
import { resolveLang, setLang, useT } from '@/i18n';
import { Button } from '@/ui/Button';
import { ErrorCard } from '@/ui/ErrorCard';
import { bootstrapToken } from './bootstrap';
import { ErrorBoundary } from './ErrorBoundary';
import { MultiTabScreen } from './MultiTabScreen';
import { applyPrefs, loadPrefs, systemMatchMedia } from './prefs';
import { startRouter } from './router';
import { getSessionStorage } from './sessionStorage';
import { Shell } from './Shell';
import { acquireTabLock } from './singleTab';
import { useDraftHeld } from '@/features/draft/useDraftHeld';
import { StaleScreen } from './StaleScreen';

type Boot =
  | { kind: 'booting' }
  | { kind: 'other-tab' }
  | { kind: 'no-token' }
  | { kind: 'error'; code: ErrorCode; detail: unknown }
  | { kind: 'ready' };

export function App() {
  const t = useT();
  const [boot, setBoot] = useState<Boot>({ kind: 'booting' });
  const stale = useStore(sessionStore, (s) => s.stale);
  const hasDraft = useStore(draftStatusStore, (s) => s.hasDraft);
  const draftHeld = useDraftHeld();
  const [closed, setClosed] = useState(false);
  const leave = useLeaveGuard({ beforeUnload: !closed });
  const [closeFailure, setCloseFailure] = useState<{ code: ErrorCode; detail: unknown } | null>(null);

  // A failed shutdown changes nothing else: the server is still up, so the app stays as it was and says so.
  const shutdown = () => {
    setCloseFailure(null);
    getApi()
      .shutdown()
      .then(
        () => setClosed(true),
        (error: unknown) =>
          setCloseFailure(
            error instanceof ApiError ? { code: error.code, detail: error.detail } : { code: 'internal', detail: String(error) },
          ),
      );
  };

  useEffect(() => {
    let cancelled = false;
    let stopHeartbeat: () => void = () => undefined;
    let stopRouter: () => void = () => undefined;
    let stopAutosave: () => void = () => undefined;

    void (async () => {
      // The tab lock comes first (F26): a second tab on the token-stripped URL must say "open in another tab",
      // not "open the printed address".
      if ((await acquireTabLock()) === 'other') {
        if (!cancelled) setBoot({ kind: 'other-tab' });
        return;
      }
      if (cancelled) return;
      setLang(resolveLang(null, navigator.language)); // until the prefs say otherwise
      const token = bootstrapToken(window.location, getSessionStorage(), window.history, import.meta.env.DEV);
      if (token === null) {
        setBoot({ kind: 'no-token' });
        return;
      }
      const session = sessionStore.getState();
      const api = createApi({
        token,
        onStale: () => session.setStale('restarted'),
        onBusy: () => session.setBusy(true),
      });
      setApi(api);

      const prefs = await loadPrefs(api);
      if (cancelled) return;
      applyPrefs(prefs, resolveLang(prefs.lang, navigator.language), systemMatchMedia());
      try {
        const [system, current] = await Promise.all([api.system(), api.session()]);
        if (cancelled) return;
        session.setSystem(system);
        session.setModel(current.loaded ? current : null);
      } catch (error) {
        if (cancelled) return;
        // A 403 already moved the app to the stale screen; anything else is shown as an error card.
        if (error instanceof ApiError && error.code !== 'forbidden') {
          setBoot({ kind: 'error', code: error.code, detail: error.detail });
        } else if (!(error instanceof ApiError)) {
          setBoot({ kind: 'error', code: 'internal', detail: String(error) });
        }
        return;
      }
      stopHeartbeat = startHeartbeat(api, () => session.setStale('down'), {
        onUp: () => {
          if (sessionStore.getState().stale === 'down') session.setStale('none');
        },
      });
      stopRouter = startRouter();
      stopAutosave = startAutosave({
        api,
        basket: basketStore,
        session: sessionStore,
        prompts: () => {
          const prompt = compareStore.getState().lastPrompt;
          return prompt === null ? [] : [prompt];
        },
      });
      setBoot({ kind: 'ready' });
    })();

    return () => {
      cancelled = true;
      stopHeartbeat();
      stopRouter();
      stopAutosave();
    };
  }, []);

  if (boot.kind === 'other-tab') return <MultiTabScreen />;
  if (closed) {
    return (
      <main className="stale-screen" role="alert">
        <h1>{t('app.closed')}</h1>
        {hasDraft && <p>{t(draftHeld ? 'app.closed.draft.held' : 'app.closed.draft')}</p>}
      </main>
    );
  }
  if (stale !== 'none') return <StaleScreen kind={stale} hasDraft={hasDraft} />;
  if (boot.kind === 'no-token') {
    return (
      <main className="stale-screen">
        <h1>{t('app.noToken')}</h1>
      </main>
    );
  }
  if (boot.kind === 'error') {
    return (
      <main className="stale-screen">
        <h1>{t('app.bootError')}</h1>
        <ErrorCard code={boot.code} detail={boot.detail} />
      </main>
    );
  }
  if (boot.kind === 'booting') {
    return (
      <main className="stale-screen" aria-busy="true">
        <p>{t('app.loading')}</p>
      </main>
    );
  }
  return (
    <ErrorBoundary>
      {closeFailure !== null && (
        <div className="close-failure">
          <p>{t('app.closeFailed')}</p>
          <ErrorCard code={closeFailure.code} detail={closeFailure.detail} />
          <Button onClick={() => setCloseFailure(null)}>{t('ui.ok')}</Button>
        </div>
      )}
      <Shell onClose={() => leave.request('close', shutdown)} />
      {leave.dialog}
    </ErrorBoundary>
  );
}
