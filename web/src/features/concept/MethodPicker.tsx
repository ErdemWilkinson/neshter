import { useId } from 'react';
import type { HuntResult } from '@/api/schemas';
import { useT, type MessageKey } from '@/i18n';

export type HuntMethod = HuntResult['method'];

const METHODS: { value: HuntMethod; label: MessageKey; note?: MessageKey }[] = [
  { value: 'attribution', label: 'concept.method.attribution', note: 'concept.method.attribution.note' },
  { value: 'activation_diff', label: 'concept.method.activation_diff' },
  { value: 'direction', label: 'concept.method.direction' },
];

/** The three ways to search; no durations are promised. */
export function MethodPicker({ value, onChange }: { value: HuntMethod; onChange: (method: HuntMethod) => void }) {
  const t = useT();
  const name = useId();
  return (
    <fieldset className="method-picker">
      <legend>{t('concept.method')}</legend>
      {METHODS.map((method) => (
        <label key={method.value} className="method-option">
          <input
            type="radio"
            name={name}
            value={method.value}
            checked={method.value === value}
            onChange={() => onChange(method.value)}
          />
          <span>{t(method.label)}</span>
          {method.note !== undefined && <small className="method-note">{t(method.note)}</small>}
        </label>
      ))}
    </fieldset>
  );
}
