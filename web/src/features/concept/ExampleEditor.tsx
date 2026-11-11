import { useId } from 'react';
import { useT } from '@/i18n';

export const MIN_EXAMPLES = 32;

/** One request per line; empty lines are ignored. */
export const parseLines = (text: string): string[] =>
  text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line !== '');

function Side({
  label,
  text,
  readOnly,
  onChange,
}: {
  label: string;
  text: string;
  readOnly: boolean;
  onChange: (text: string) => void;
}) {
  const t = useT();
  const id = useId();
  const n = parseLines(text).length;
  return (
    <div className="example-side">
      <label htmlFor={id}>{label}</label>
      <textarea id={id} rows={8} value={text} readOnly={readOnly} onChange={(event) => onChange(event.target.value)} />
      <p className={`num example-count${n < MIN_EXAMPLES ? ' example-count-low' : ''}`} aria-live="polite">
        {t('concept.examples.count', { n })}
      </p>
    </div>
  );
}

/** The two example lists (behaviour shown / not shown) with live counts; a preset's lists are read-only. */
export function ExampleEditor({
  positive,
  negative,
  readOnly,
  onPositive,
  onNegative,
}: {
  positive: string;
  negative: string;
  readOnly: boolean;
  onPositive: (text: string) => void;
  onNegative: (text: string) => void;
}) {
  const t = useT();
  return (
    <div className="example-editor">
      <div className="example-sides">
        <Side label={t('concept.examples.positive')} text={positive} readOnly={readOnly} onChange={onPositive} />
        <Side label={t('concept.examples.negative')} text={negative} readOnly={readOnly} onChange={onNegative} />
      </div>
      <p className="example-note">{t('concept.examples.note')}</p>
    </div>
  );
}
