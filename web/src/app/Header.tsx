import { useStore } from 'zustand';
import { getApi } from '@/api/currentApi';
import { unsavedCount } from '@/core/basket';
import { basketStore } from '@/core/stores/basket';
import { sessionStore } from '@/core/stores/session';
import { uiStore, type Lang } from '@/core/stores/ui';
import { plural, useT } from '@/i18n';
import { Button } from '@/ui/Button';
import { chooseLang, chooseReducedMotion, isMotionReduced, systemMatchMedia } from './prefs';

const GIB = 1024 ** 3;
const LANGS = ['tr', 'en'] as const satisfies readonly Lang[];

/** Whole-number or one-decimal GB, so the chip stays short (`31,7`). */
const gigabytes = (bytes: number): number => Math.round((bytes / GIB) * 10) / 10;

export function Header({ onClose }: { onClose: () => void }) {
  const t = useT();
  const system = useStore(sessionStore, (s) => s.system);
  const model = useStore(sessionStore, (s) => s.model);
  const unsaved = useStore(basketStore, (s) => unsavedCount(s.items, s.savedOps));
  const lang = useStore(uiStore, (s) => s.lang);
  const reducedMotion = useStore(uiStore, (s) => s.reducedMotion);

  return (
    <header className="app-header glass">
      <span className="app-name">{t('app.name')}</span>
      <ul className="chips">
        {system !== null && (
          <>
            <li className="chip">{system.gpu_name === null ? t('header.cpuOnly') : t('header.gpu', { name: system.gpu_name })}</li>
            {system.gpu_name !== null && <li className="chip num">{t('header.vram', { gb: gigabytes(system.vram_bytes) })}</li>}
            <li className="chip num">{t('header.ram', { gb: gigabytes(system.ram_bytes) })}</li>
          </>
        )}
        {model !== null && (
          <li className="chip chip-model" title={model.name}>
            {t('header.model', { name: model.name, dtype: model.dtype, device: model.device })}
          </li>
        )}
        {unsaved > 0 && <li className="chip chip-unsaved">{plural('header.unsaved', unsaved)}</li>}
      </ul>
      <div className="header-controls">
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
        <Button variant="ghost" onClick={onClose}>
          {t('header.close')}
        </Button>
      </div>
    </header>
  );
}
