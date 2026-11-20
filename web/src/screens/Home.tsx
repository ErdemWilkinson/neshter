import type { MouseEvent } from 'react';
import { isDeviceApp } from '@/api/device/bridge';
import { navigate } from '@/app/router';
import { uiStore, type Intent } from '@/core/stores/ui';
import { demoStore } from '@/features/demo/demoStore';
import { UnsavedWork } from '@/features/draft/UnsavedWork';
import { ImportRecipe } from '@/features/recipe/ImportRecipe';
import { useT } from '@/i18n';

function choose(intent: Exclude<Intent, null>): void {
  demoStore.getState().stop();
  uiStore.getState().setIntent(intent);
  navigate('models');
}

/** The example: download and load the sample model, then Explore runs the text and measures on its own. */
function tryExample(): void {
  uiStore.getState().setIntent('inspect');
  demoStore.getState().start();
  navigate('models');
}

interface Row {
  key: string;
  title: string;
  hint: string;
  onClick: () => void;
  demo?: boolean;
}

export function Home() {
  const t = useT();
  const device = isDeviceApp();
  const rows: Row[] = [
    ...(device ? [] : [{ key: 'demo', title: t('home.demo'), hint: t('home.demo.hint'), onClick: tryExample, demo: true }]),
    device
      ? { key: 'chat', title: t('home.chat'), hint: t('home.chat.hint'), onClick: () => choose('chat') }
      : { key: 'reduce', title: t('home.reduce'), hint: t('home.reduce.hint'), onClick: () => choose('reduce') },
    { key: 'inspect', title: t('home.inspect'), hint: t('home.inspect.hint'), onClick: () => choose('inspect') },
  ];
  return (
    <section className="home-screen">
      <div className="home-hero">
        <p className="home-kicker num">{t('home.kicker')}</p>
        <p className="home-headline" aria-hidden="true">
          {t('home.headline')}
        </p>
        <p className="home-promise num">{t('home.promise')}</p>
      </div>
      <h1 className="home-question">{t('home.title')}</h1>
      <ol className="intent-rows">
        {rows.map((row, i) => (
          <li key={row.key}>
            <button type="button" className={`intent-row${row.demo ? ' is-demo' : ''}`} onClick={row.onClick}>
              <span className="intent-num num" aria-hidden="true">
                {String(i + 1).padStart(2, '0')}
              </span>
              <span className="intent-text">
                <span className="intent-title">{row.title}</span>
                <span className="intent-hint">{row.hint}</span>
              </span>
              <span className="intent-arrow" aria-hidden="true">
                →
              </span>
            </button>
          </li>
        ))}
      </ol>
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
