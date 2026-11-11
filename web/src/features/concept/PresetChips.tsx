import type { Concept } from '@/api/schemas';
import { uiStore, type Lang } from '@/core/stores/ui';
import { useT } from '@/i18n';
import { Segmented } from '@/ui/Segmented';

/** Not a valid concept id (the server's ids are slugs), so it can never clash with a preset. */
export const CUSTOM = '+custom';

export const conceptName = (concept: Concept, lang: Lang): string => (lang === 'en' ? concept.name_en : concept.name_tr);

/** The presets in the order the API gives them, then `+ Kendi kavramın`. */
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
  const lang = uiStore.getState().lang;
  const options = [
    ...concepts.map((concept) => ({ value: concept.id, label: conceptName(concept, lang) })),
    { value: CUSTOM, label: t('concept.custom') },
  ];
  return <Segmented label={t('concept.presets')} options={options} value={value} onChange={onChange} />;
}
