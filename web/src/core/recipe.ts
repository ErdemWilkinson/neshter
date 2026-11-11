// Strict mirror of neshter/recipe.py `Recipe`: recipes come from other people, so unknown fields are rejected.
import { newBasketItem, savableOps, type BasketItem } from './basket';
import { clip } from './draft';
import { DirectionOpSchema, HeadOpSchema, isDirectionOp, isHeadOp, parseOp } from './ops';
import { z } from './zod';

/** Same as `recipe.MAX_RECIPE_BYTES`. */
export const MAX_RECIPE_BYTES = 5 * 1024 * 1024;
const MAX_TEXT = 256;
const RECIPE_TEXT = /^[A-Za-z0-9_.-]{1,64}$/;

export const RecipeSchema = z.strictObject({
  schema_version: z.literal(1),
  neshter_version: z.string().max(MAX_TEXT),
  source_model: z.strictObject({
    name: z.string().max(MAX_TEXT),
    sha256: z.string().regex(/^[0-9a-f]{64}$/),
  }),
  concept: z.string().regex(RECIPE_TEXT).nullable(),
  method: z.string().regex(RECIPE_TEXT),
  seed: z.number().int(),
  dataset_sha256: z.string().nullable(),
  head_mode: z.literal('zero'),
  interventions: z.array(z.discriminatedUnion('type', [HeadOpSchema, DirectionOpSchema])),
  created_at: z.iso.datetime({ offset: true }),
});

export type Recipe = z.infer<typeof RecipeSchema>;

export type RecipeParseResult =
  | { ok: true; recipe: Recipe }
  | { ok: false; error: 'json' | 'schema' | 'too_large' };

export function parseRecipeText(text: string): RecipeParseResult {
  if (new TextEncoder().encode(text).length > MAX_RECIPE_BYTES) return { ok: false, error: 'too_large' };
  let json: unknown;
  try {
    json = JSON.parse(text);
  } catch {
    return { ok: false, error: 'json' };
  }
  const parsed = RecipeSchema.safeParse(json);
  return parsed.success ? { ok: true, recipe: parsed.data } : { ok: false, error: 'schema' };
}

export function recipeToItems(recipe: Recipe, now: string): BasketItem[] {
  return recipe.interventions.map((op) =>
    newBasketItem(parseOp(op), { origin: 'import', concept: recipe.concept, addedAt: now }),
  );
}

/**
 * The recipe for the savable ops of `items`. The recipe format only knows head and direction ops, so a savable
 * kind registered later would need a schema bump; it is left out here rather than written as something
 * `recipe.py` would reject.
 */
export function itemsToRecipe(
  items: readonly BasketItem[],
  model: { name: string; sha256: string },
  concept: string | null,
  now: string,
): Recipe {
  return {
    schema_version: 1,
    neshter_version: 'web',
    source_model: { name: clip(model.name, MAX_TEXT), sha256: model.sha256 },
    concept,
    method: 'ui',
    seed: 0,
    dataset_sha256: null,
    head_mode: 'zero',
    interventions: savableOps(items).flatMap((op) => (isHeadOp(op) || isDirectionOp(op) ? [op] : [])),
    created_at: now,
  };
}
