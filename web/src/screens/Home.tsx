import type { MouseEvent } from 'react';
import { navigate } from '@/app/router';
import { uiStore, type Intent } from '@/core/stores/ui';
import { UnsavedWork } from '@/features/draft/UnsavedWork';
import { ImportRecipe } from '@/features/recipe/ImportRecipe';
import { useT } from '@/i18n';

function choose(intent: Exclude<Intent, null>): void {
  uiStore.getState().setIntent(intent);
  navigate('models');
}

export function Home() {
  const t = useT();
  return (
    <section className="home-screen">
      <h1>{t('home.title')}</h1>
      <div className="intent-cards">
        <button type="button" className="intent-card glass" onClick={() => choose('reduce')}>
          <span className="intent-title">{t('home.reduce')}</span>
          <span className="intent-hint">{t('home.reduce.hint')}</span>
        </button>
        <button type="button" className="intent-card glass" onClick={() => choose('inspect')}>
          <span className="intent-title">{t('home.inspect')}</span>
          <span className="intent-hint">{t('home.inspect.hint')}</span>
        </button>
      </div>
      <a
        className="expert-link"
        href="#/models"
        onClick={(event: MouseEvent) => {
          event.preventDefault();
          choose('expert');
        }}
      >
        {t('home.expert')}
      </a>
      <ImportRecipe />
      <UnsavedWork />
    </section>
  );
}
