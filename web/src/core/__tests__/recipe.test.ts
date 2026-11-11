import { MAX_RECIPE_BYTES, RecipeSchema, itemsToRecipe, parseRecipeText, recipeToItems } from '../recipe';
import { isOpaque } from '../ops';
import { registerFakeExploreOp } from './fixtures/fakeExploreOp';
import { NOW, SHA, directionOp, head, headOp, item } from './fixtures/items';

beforeAll(registerFakeExploreOp);

const model = { name: 'tiny-model', sha256: SHA };
const validRecipe = () => ({
  schema_version: 1,
  neshter_version: '0.1.0',
  source_model: { name: 'tiny-model', sha256: SHA },
  concept: 'code',
  method: 'activation_diff',
  seed: 0,
  dataset_sha256: null,
  head_mode: 'zero',
  interventions: [headOp(0, 1), directionOp([1, 2])],
  created_at: '2026-11-11T08:22:24.203903+00:00',
});

test('itemsToRecipe keeps only savable ops, sets neshter_version and created_at, and parses with RecipeSchema', () => {
  const items = [head(0, 1, { concept: 'code' }), item(directionOp([1, 0, 2])), item({ type: 'test_explore', src: 'h:0:1' }), item({ type: 'edge_read_v9' })];
  const recipe = itemsToRecipe(items, model, 'code', NOW);
  expect(recipe).toMatchObject({
    schema_version: 1,
    neshter_version: 'web',
    created_at: NOW,
    method: 'ui',
    seed: 0,
    dataset_sha256: null,
    head_mode: 'zero',
    concept: 'code',
    source_model: model,
  });
  expect(recipe.interventions.map((op) => op.type)).toEqual(['head', 'direction']);
  expect(RecipeSchema.safeParse(JSON.parse(JSON.stringify(recipe))).success).toBe(true);
});

test('itemsToRecipe clips a long model name to the recipe limit', () => {
  const recipe = itemsToRecipe([head(0, 1)], { name: 'n'.repeat(400), sha256: SHA }, null, NOW);
  expect(recipe.source_model.name).toHaveLength(256);
  expect(RecipeSchema.safeParse(recipe).success).toBe(true);
});

test('parseRecipeText accepts what recipe.py writes (offset timestamps) and round-trips to items', () => {
  const parsed = parseRecipeText(JSON.stringify(validRecipe()));
  if (!parsed.ok) throw new Error('expected ok');
  const items = recipeToItems(parsed.recipe, NOW);
  expect(items.map((i) => [i.origin, i.concept, i.semantics, i.addedAt])).toEqual([
    ['import', 'code', 'zero', NOW],
    ['import', 'code', 'direction', NOW],
  ]);
  expect(items.some((i) => isOpaque(i.op))).toBe(false);
});

test('parseRecipeText rejects unknown fields and > 5 MiB', () => {
  expect(parseRecipeText(JSON.stringify({ ...validRecipe(), extra: 1 }))).toEqual({ ok: false, error: 'schema' });
  expect(parseRecipeText(JSON.stringify({ ...validRecipe(), interventions: [{ ...headOp(0, 1), extra: 1 }] }))).toEqual({ ok: false, error: 'schema' });
  expect(parseRecipeText('{not json')).toEqual({ ok: false, error: 'json' });
  const body = JSON.stringify(validRecipe());
  const exact = body + ' '.repeat(MAX_RECIPE_BYTES - body.length);
  expect(new TextEncoder().encode(exact).length).toBe(MAX_RECIPE_BYTES);
  expect(parseRecipeText(exact).ok).toBe(true);
  expect(parseRecipeText(`${exact} `)).toEqual({ ok: false, error: 'too_large' });
});

test('the size limit counts UTF-8 bytes, not characters', () => {
  const body = JSON.stringify(validRecipe());
  const padded = body + 'ç'.repeat(Math.ceil((MAX_RECIPE_BYTES - body.length) / 2) + 1); // 2 bytes each, past the limit
  expect(padded.length).toBeLessThan(MAX_RECIPE_BYTES);
  expect(parseRecipeText(padded)).toEqual({ ok: false, error: 'too_large' });
});

test('parseRecipeText mirrors recipe.py field rules', () => {
  const bad = (over: Record<string, unknown>) => parseRecipeText(JSON.stringify({ ...validRecipe(), ...over }));
  expect(bad({ schema_version: 2 }).ok).toBe(false);
  expect(bad({ head_mode: 'mean' }).ok).toBe(false);
  expect(bad({ concept: 'has space' }).ok).toBe(false);
  expect(bad({ concept: null }).ok).toBe(true);
  expect(bad({ method: '' }).ok).toBe(false);
  expect(bad({ source_model: { name: 'x', sha256: 'ABC' } }).ok).toBe(false);
  expect(bad({ created_at: '2026-11-11T08:22:24' }).ok).toBe(false); // naive datetime
  expect(bad({ interventions: [{ type: 'edge_read_v9' }] }).ok).toBe(false);
  expect(bad({ interventions: [directionOp([0, 0])] }).ok).toBe(false); // zero vector
  expect(bad({ interventions: [{ ...headOp(0, 1), layer: -1 }] }).ok).toBe(false);
  expect(bad({ seed: 1.5 }).ok).toBe(false);
});
