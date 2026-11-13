import { useId } from 'react';
import { useT } from '@/i18n';

/**
 * `Tekrarlı süpürme`: after the found direction is removed, the changed model is run again; if the concept still
 * separates at some layer, that layer's direction is removed too (up to `value` directions). 1 = off.
 */
export function SweepField({ value, onChange }: { value: number; onChange: (n: number) => void }) {
  const t = useT();
  const id = useId();
  const hintId = useId();
  return (
    <div className="sweep-field">
      <label htmlFor={id}>{t('concept.sweep.label')}</label>
      <select id={id} value={value} aria-describedby={hintId} onChange={(event) => onChange(Number(event.target.value))}>
        {Array.from({ length: 8 }, (_, i) => i + 1).map((n) => (
          <option key={n} value={n}>
            {n === 1 ? t('concept.sweep.off') : n}
          </option>
        ))}
      </select>
      <p id={hintId} className="field-hint">
        {t('concept.sweep.hint')}
      </p>
    </div>
  );
}
