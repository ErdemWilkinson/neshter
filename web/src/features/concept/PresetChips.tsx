import { useId } from 'react';
import type { Concept } from '@/api/schemas';
import { uiStore, type Lang } from '@/core/stores/ui';
import { useT } from '@/i18n';

/** Not a valid concept id (the server's ids are slugs), so it can never clash with a preset. */
export const CUSTOM = '+custom';

export const conceptName = (concept: Concept, lang: Lang): string => (lang === 'en' ? concept.name_en : concept.name_tr);

/**
 * The presets in the order the API gives them, then `+ Kendi kavramın`, as cards: the name, one of its example
 * sentences (what the concept looks like in text) and a mark on the safety one. Native radios underneath, so the
 * arrow keys and the spoken role are the browser's.
 */
export function PresetChips({
  concepts,
  value,
  onChange,
}: {
  concepts: readonly Concept[];
  value: string;
  onChange: (id: string) => void;
}) {
  const t = useT();
  const name = useId();
  const lang = uiStore.getState().lang;
  const cards = [
    ...concepts.map((concept) => ({
      value: concept.id,
      label: conceptName(concept, lang),
      example: concept.positive[0] ?? null,
      safety: concept.safety_ablation,
    })),
    { value: CUSTOM, label: t('concept.custom'), example: null, safety: false },
  ];
  return (
    <fieldset className="preset-cards">
      <legend className="sr-only">{t('concept.presets')}</legend>
      {cards.map((card) => (
        <label key={card.value} className="preset-card">
          <input
            type="radio"
            name={name}
            value={card.value}
            checked={card.value === value}
            // The card's name is the concept; the example sentence and the badge are visual help.
            aria-label={card.label}
            onChange={() => onChange(card.value)}
          />
          <span className="preset-name">{card.label}</span>
          {card.example !== null ? (
            <span className="preset-example">{t('concept.card.example', { text: card.example })}</span>
          ) : (
            card.value === CUSTOM && <span className="preset-example">{t('concept.card.custom')}</span>
          )}
          {card.safety && <span className="badge badge-safety">{t('concept.card.safety')}</span>}
        </label>
      ))}
    </fieldset>
  );
}
