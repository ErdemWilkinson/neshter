import { useEffect, useRef, type ReactNode } from 'react';
import { useStore } from 'zustand';
import { getApi } from '@/api/currentApi';
import { isDeviceApp } from '@/api/device/bridge';
import { unsavedCount } from '@/core/basket';
import { basketStore } from '@/core/stores/basket';
import { sessionStore } from '@/core/stores/session';
import { uiStore, type Lang } from '@/core/stores/ui';
import { plural, useT } from '@/i18n';
import { Button } from '@/ui/Button';
import { chooseLang, chooseReducedMotion, isMotionReduced, systemMatchMedia } from './prefs';
import { navigate } from './router';

const GIB = 1024 ** 3;
const LANGS = ['tr', 'en'] as const satisfies readonly Lang[];

/** Whole-number or one-decimal GB, so the chip stays short (`31,7`). */
const gigabytes = (bytes: number): number => Math.round((bytes / GIB) * 10) / 10;

/**
 * One bar: name and model on the left, the steps in the middle (`nav`), the basket and settings on the right. The
 * basket button says how many cuts are waiting and opens the comparison, where they are tried and saved.
 */
/** "NVIDIA GeForce RTX 4050 Laptop GPU" -> "RTX 4050 Laptop": the vendor words only cost header room (full name on hover). */
export const shortGpu = (name: string): string =>
  name.replace(/^NVIDIA\s+/i, '').replace(/^GeForce\s+/i, '').replace(/\s+GPU$/i, '').trim() || name;

export function Header({ onClose, nav }: { onClose: () => void; nav?: ReactNode }) {
  const t = useT();
  const system = useStore(sessionStore, (s) => s.system);
  const model = useStore(sessionStore, (s) => s.model);
  const unsaved = useStore(basketStore, (s) => unsavedCount(s.items, s.savedOps));
  const inBasket = useStore(basketStore, (s) => s.items.length);
  const lang = useStore(uiStore, (s) => s.lang);
  const reducedMotion = useStore(uiStore, (s) => s.reducedMotion);
  const bar = useRef<HTMLElement>(null);

  // The header is sticky and wraps to one, two or three rows; sticky panels below sit under its real height.
  useEffect(() => {
    const el = bar.current;
    if (el === null || typeof ResizeObserver === 'undefined') return;
    const root = document.documentElement;
    const publish = () => root.style.setProperty('--header-h', `${Math.ceil(el.getBoundingClientRect().height)}px`);
    publish();
    const observer = new ResizeObserver(publish);
    observer.observe(el);
    return () => {
      observer.disconnect();
      root.style.removeProperty('--header-h');
    };
  }, []);

  return (
    <header ref={bar} className="app-header glass">
      <div className="header-brand">
        <span className="app-name">{t('app.name')}</span>
        {model !== null && (
          <span className="model-chip" title={model.name}>
            <span className="live-dot" aria-hidden="true" />
            {t('header.model', { name: model.name, dtype: model.dtype, device: model.device })}
          </span>
        )}
      </div>
      {nav}
      <ul className="chips">
        {system !== null && (
          <>
            <li className="chip" title={system.gpu_name ?? undefined}>
              {system.gpu_name === null ? t('header.cpuOnly') : t('header.gpu', { name: shortGpu(system.gpu_name) })}
            </li>
            {system.gpu_name !== null && <li className="chip num">{t('header.vram', { gb: gigabytes(system.vram_bytes) })}</li>}
            <li className="chip num">{t('header.ram', { gb: gigabytes(system.ram_bytes) })}</li>
          </>
        )}
      </ul>
      <div className="header-controls">
        {inBasket > 0 && (
          <Button className={`basket-button${unsaved > 0 ? ' has-unsaved' : ''}`} onClick={() => navigate('compare')}>
            {unsaved > 0 ? plural('header.unsaved', unsaved) : plural('header.basket', inBasket)}
          </Button>
        )}
        <label className="control">
          <span>{t('header.lang')}</span>
          <select value={lang} onChange={(event) => chooseLang(getApi(), event.target.value === 'en' ? 'en' : 'tr')}>
            {LANGS.map((code) => (
              <option key={code} value={code}>
                {code === 'tr' ? t('header.lang.tr') : t('header.lang.en')}
              </option>
            ))}
          </select>
        </label>
        <label className="control">
          <input
            type="checkbox"
            checked={isMotionReduced(reducedMotion, systemMatchMedia())}
            onChange={(event) => chooseReducedMotion(getApi(), event.target.checked, systemMatchMedia())}
          />
          <span>{t('header.motion')}</span>
        </label>
        {!isDeviceApp() && (
          <Button variant="ghost" title={t('header.close.hint')} onClick={onClose}>
            {t('header.close')}
          </Button>
        )}
      </div>
    </header>
  );
}
