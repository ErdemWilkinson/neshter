import { useEffect, useRef, type ComponentType } from 'react';
import { useStore } from 'zustand';
import { uiStore, type Route } from '@/core/stores/ui';
import { DraftNotice } from '@/features/draft/DraftNotice';
import { RestorePrompt } from '@/features/draft/RestorePrompt';
import { useT } from '@/i18n';
import { LiveRegion, announce } from '@/ui/LiveRegion';
import { BusyBanner } from './BusyBanner';
import { DeviceBanner } from './DeviceBanner';
import { ErrorBoundary } from './ErrorBoundary';
import { Header } from './Header';
import { STEP_LABELS, Stepper } from './Stepper';

function Placeholder({ route }: { route: Route }) {
  const t = useT();
  return <h1>{t(STEP_LABELS[route])}</h1>;
}

const placeholder = (route: Route): ComponentType =>
  function PlaceholderScreen() {
    return <Placeholder route={route} />;
  };

/**
 * The route table. Tasks 9-14 each replace their own entries here with the real screen; every key of the Route
 * union must have an entry (the type enforces it, `Shell.test` pins it at run time).
 */
export const SCREENS: Record<Route, ComponentType> = {
  home: placeholder('home'),
  models: placeholder('models'),
  explore: placeholder('explore'),
  concept: placeholder('concept'),
  compare: placeholder('compare'),
  save: placeholder('save'),
};

export function Shell({ onClose }: { onClose: () => void }) {
  const t = useT();
  const route = useStore(uiStore, (s) => s.route);
  const Screen = SCREENS[route];
  const mainRef = useRef<HTMLElement>(null);
  const shownRoute = useRef(route);

  // A route change moves focus to the new screen and says its name; the first render leaves focus alone.
  useEffect(() => {
    if (shownRoute.current === route) return;
    shownRoute.current = route;
    mainRef.current?.focus();
    announce(t(STEP_LABELS[route]));
  }, [route, t]);

  return (
    <div className="shell">
      <button type="button" className="skip-link" onClick={() => mainRef.current?.focus()}>
        {t('app.skip')}
      </button>
      <Header onClose={onClose} />
      <Stepper />
      <DeviceBanner />
      <BusyBanner />
      <DraftNotice />
      <RestorePrompt />
      <main className="screen" ref={mainRef} tabIndex={-1}>
        <ErrorBoundary key={route}>
          <Screen />
        </ErrorBoundary>
      </main>
      <LiveRegion />
    </div>
  );
}
