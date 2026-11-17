import { Fragment } from 'react';
import { useStore } from 'zustand';
import type { Route } from '@/core/stores/ui';
import { uiStore } from '@/core/stores/ui';
import { useT, type MessageKey } from '@/i18n';
import { isDeviceApp } from '@/api/device/bridge';

export const STEP_LABELS: Record<Route, MessageKey> = {
  home: 'app.step.home',
  models: 'app.step.models',
  concept: 'app.step.concept',
  explore: 'app.step.explore',
  compare: 'app.step.compare',
  save: 'app.step.save',
  chat: 'app.step.chat',
};

// Model -> (Davranışı bul | Ağ haritası) -> Karşılaştır -> Kaydet: the middle step has two entrances.
const STEPS: readonly (readonly Route[])[] = [['models'], ['concept', 'explore'], ['compare'], ['save']];

// The phone keeps the steps that run on the device: no behaviour search (gradients), no saved model files.
const PHONE_STEPS: readonly (readonly Route[])[] = [['models'], ['chat', 'explore'], ['compare']];

export function Stepper() {
  const t = useT();
  const route = useStore(uiStore, (s) => s.route);
  if (route === 'home') return null;
  return (
    <nav className="stepper" aria-label={t('app.stepper.label')}>
      <ol>
        {(isDeviceApp() ? PHONE_STEPS : STEPS).map((entrances) => (
          <li key={entrances.join('|')} className={entrances.includes(route) ? 'is-current' : undefined}>
            {entrances.map((entrance, i) => (
              <Fragment key={entrance}>
                {i > 0 && (
                  <span className="stepper-or" aria-hidden="true">
                    {t('app.step.or')}
                  </span>
                )}
                <a href={`#/${entrance}`} aria-current={entrance === route ? 'step' : undefined}>
                  {t(STEP_LABELS[entrance])}
                </a>
              </Fragment>
            ))}
          </li>
        ))}
      </ol>
    </nav>
  );
}
