import { useStore } from 'zustand';
import type { Route } from '@/core/stores/ui';
import { uiStore } from '@/core/stores/ui';
import { useT, type MessageKey } from '@/i18n';

export const STEP_LABELS: Record<Route, MessageKey> = {
  home: 'app.step.home',
  models: 'app.step.models',
  concept: 'app.step.concept',
  explore: 'app.step.explore',
  compare: 'app.step.compare',
  save: 'app.step.save',
};

// Model -> (Davranışı bul | Ağ haritası) -> Karşılaştır -> Kaydet: the middle step has two entrances.
const STEPS: readonly (readonly Route[])[] = [['models'], ['concept', 'explore'], ['compare'], ['save']];

export function Stepper() {
  const t = useT();
  const route = useStore(uiStore, (s) => s.route);
  if (route === 'home') return null;
  return (
    <nav className="stepper" aria-label={t('app.stepper.label')}>
      <ol>
        {STEPS.map((entrances) => (
          <li key={entrances.join('|')} className={entrances.includes(route) ? 'is-current' : undefined}>
            {entrances.map((entrance) => (
              <a key={entrance} href={`#/${entrance}`} aria-current={entrance === route ? 'step' : undefined}>
                {t(STEP_LABELS[entrance])}
              </a>
            ))}
          </li>
        ))}
      </ol>
    </nav>
  );
}
