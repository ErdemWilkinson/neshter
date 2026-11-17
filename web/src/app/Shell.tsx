import { useEffect, useRef, type ComponentType } from 'react';
import { useStore } from 'zustand';
import { uiStore, type Route } from '@/core/stores/ui';
import { DraftNotice } from '@/features/draft/DraftNotice';
import { RestorePrompt } from '@/features/draft/RestorePrompt';
import { Compare } from '@/screens/Compare';
import { Concept } from '@/screens/Concept';
import { Explore } from '@/screens/Explore';
import { Home } from '@/screens/Home';
import { Models } from '@/screens/Models';
import { Chat } from '@/screens/Chat';
import { Save } from '@/screens/Save';
import { useT } from '@/i18n';
import { LiveRegion, announce } from '@/ui/LiveRegion';
import { BasketToast } from './BasketToast';
import { BusyBanner } from './BusyBanner';
import { DeviceBanner } from './DeviceBanner';
import { ErrorBoundary } from './ErrorBoundary';
import { Header } from './Header';
import { STEP_LABELS, Stepper } from './Stepper';

/**
 * The route table: every key of the Route union must have an entry (the type enforces it, `Shell.test` pins it at
 * run time).
 */
export const SCREENS: Record<Route, ComponentType> = {
  home: Home,
  models: Models,
  explore: Explore,
  concept: Concept,
  compare: Compare,
  save: Save,
  chat: Chat,
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
      <Header onClose={onClose} nav={<Stepper />} />
      <DeviceBanner />
      <BusyBanner />
      <DraftNotice />
      <RestorePrompt />
      <main className="screen" ref={mainRef} tabIndex={-1}>
        <ErrorBoundary key={route}>
          <Screen />
        </ErrorBoundary>
      </main>
      <BasketToast />
      <LiveRegion />
    </div>
  );
}
