import { useEffect, useState } from 'react';
import { useStore } from 'zustand';
import { createApi } from '@/api/client';
import { getApi, setApi } from '@/api/currentApi';
import { ApiError, type ErrorCode } from '@/api/errors';
import { startHeartbeat } from '@/api/heartbeat';
import { unsavedCount } from '@/core/basket';
import { basketStore } from '@/core/stores/basket';
import { sessionStore } from '@/core/stores/session';
import { resolveLang, setLang, useT } from '@/i18n';
import { ErrorCard } from '@/ui/ErrorCard';
import { bootstrapToken } from './bootstrap';
import { ErrorBoundary } from './ErrorBoundary';
import { applyPrefs, loadPrefs, systemMatchMedia } from './prefs';
import { startRouter } from './router';
import { getSessionStorage } from './sessionStorage';
import { Shell } from './Shell';
import { StaleScreen } from './StaleScreen';

type Boot =
  | { kind: 'booting' }
  | { kind: 'no-token' }
  | { kind: 'error'; code: ErrorCode; detail: unknown }
  | { kind: 'ready' };

/** "Kapat": stop the server, then show the closed screen (the leave dialog for unsaved cuts is Task 8's). */
function closeApp(): void {
  void getApi()
    .shutdown()
    .catch(() => undefined)
    .finally(() => sessionStore.getState().setStale('down'));
}

export function App() {
  const t = useT();
  const [boot, setBoot] = useState<Boot>({ kind: 'booting' });
  const stale = useStore(sessionStore, (s) => s.stale);
  const hasDraft = useStore(basketStore, (s) => unsavedCount(s.items, s.savedOps) > 0);

  useEffect(() => {
    let cancelled = false;
    let stopHeartbeat: () => void = () => undefined;
    let stopRouter: () => void = () => undefined;

    void (async () => {
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
      setBoot({ kind: 'ready' });
    })();

    return () => {
      cancelled = true;
      stopHeartbeat();
      stopRouter();
    };
  }, []);

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
      <Shell onClose={closeApp} />
    </ErrorBoundary>
  );
}
