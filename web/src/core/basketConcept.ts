import type { Concept } from '@/api/schemas';
import type { BasketItem } from './basket';

export type ConceptInfo = Concept;

/**
 * The concept a basket stands for (recipe/export label) and the suite category that tests it.
 * `refusal` wins (it carries the safety label); otherwise every item must share one concept, else `null`
 * (mixed, a manual pick next to a concept item, or an empty basket). `suiteTarget` is the matching concept's
 * `suite_target` (`null` for `tone`, which has none, and when there is no matching concept).
 */
export function basketConcept(
  items: readonly BasketItem[],
  concepts: readonly ConceptInfo[],
): { conceptId: string | null; suiteTarget: string | null } {
  const first = items[0]?.concept ?? null;
  const conceptId = items.some((item) => item.concept?.toLowerCase() === 'refusal')
    ? 'refusal'
    : first !== null && items.every((item) => item.concept === first)
      ? first
      : null;
  const match = concepts.find((c) => conceptId !== null && c.id.toLowerCase() === conceptId.toLowerCase());
  return { conceptId, suiteTarget: match?.suite_target ?? null };
}
