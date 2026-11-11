import { useMemo, useState, type ChangeEvent } from 'react';
import { useStore } from 'zustand';
import { navigate } from '@/app/router';
import { isDirectionOp, isHeadOp } from '@/core/ops';
import { MAX_RECIPE_BYTES, parseRecipeText, recipeToItems, type RecipeParseResult } from '@/core/recipe';
import { sessionStore } from '@/core/stores/session';
import { useT } from '@/i18n';
import { Button } from '@/ui/Button';
import { pendingImportStore } from './pendingImport';
import { RecipeApply } from './RecipeApply';
import { isUnverifiedDirection } from './unverified';

type Failure = Extract<RecipeParseResult, { ok: false }>['error'];

function RecipePreview() {
  const t = useT();
  const recipe = useStore(pendingImportStore, (s) => s.recipe);
  const items = useMemo(() => (recipe === null ? [] : recipeToItems(recipe, new Date().toISOString())), [recipe]);
  if (recipe === null) return null;
  const heads = items.filter((item) => isHeadOp(item.op)).length;
  const directions = items.filter((item) => isDirectionOp(item.op)).length;
  return (
    <section className="recipe-preview glass" aria-label={t('recipe.title')}>
      <h2>{t('recipe.title')}</h2>
      <p>{t('recipe.source', { name: recipe.source_model.name })}</p>
      <p className="num">{t('recipe.count', { head: heads, direction: directions })}</p>
      <ul aria-label={t('recipe.items')}>
        {items.map((item) => (
          <li key={item.id}>
            {isHeadOp(item.op)
              ? t('recipe.item.head', { layer: item.op.layer, head: item.op.head })
              : isDirectionOp(item.op)
                ? t('recipe.item.direction', { d: item.op.d_model })
                : t('recipe.item.other', { type: item.op.type })}
            {isUnverifiedDirection(item) && <span className="badge badge-unverified">{t('vocab.badge.unverified_direction')}</span>}
          </li>
        ))}
      </ul>
    </section>
  );
}

const FAILURE_KEY = {
  json: 'recipe.error.json',
  schema: 'recipe.error.schema',
  too_large: 'recipe.error.too_large',
} as const satisfies Record<Failure, string>;

/**
 * Home's recipe import: a `.json` file is parsed (nothing is sent to the server yet) and previewed. The recipe is
 * kept in memory until a model is loaded; `LoadScreen` applies it after the load, or `Sepete ekle` applies it to
 * the model that is already loaded.
 */
export function ImportRecipe() {
  const t = useT();
  const pending = useStore(pendingImportStore, (s) => s.recipe !== null);
  const modelLoaded = useStore(sessionStore, (s) => s.model !== null);
  const [failure, setFailure] = useState<Failure | null>(null);
  const [applying, setApplying] = useState(false);

  const fail = (reason: Failure) => {
    pendingImportStore.getState().set(null);
    setFailure(reason);
  };

  const onFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const input = event.currentTarget;
    const file = input.files?.[0];
    input.value = ''; // the same file can be chosen again
    if (file === undefined) return;
    setApplying(false);
    if (file.size > MAX_RECIPE_BYTES) return fail('too_large');
    const parsed = parseRecipeText(await file.text());
    if (!parsed.ok) return fail(parsed.error);
    setFailure(null);
    pendingImportStore.getState().set(parsed.recipe);
  };

  return (
    <div className="import-recipe">
      <label className="btn btn-secondary file-button">
        {t('home.import')}
        <input type="file" accept=".json,application/json" className="sr-only" onChange={(event) => void onFile(event)} />
      </label>
      {failure !== null && (
        <p role="alert" className="import-error">
          {t(FAILURE_KEY[failure])}
        </p>
      )}
      <RecipePreview />
      {pending && !applying && (
        <div className="recipe-actions">
          {modelLoaded ? (
            <Button variant="primary" onClick={() => setApplying(true)}>
              {t('recipe.add')}
            </Button>
          ) : (
            <>
              <p>{t('recipe.needModel')}</p>
              <Button variant="primary" onClick={() => navigate('models')}>
                {t('recipe.pickModel')}
              </Button>
            </>
          )}
          <Button variant="ghost" onClick={() => pendingImportStore.getState().set(null)}>
            {t('recipe.discard')}
          </Button>
        </div>
      )}
      {applying && <RecipeApply onSettled={() => setApplying(false)} />}
    </div>
  );
}
