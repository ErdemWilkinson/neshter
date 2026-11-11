import { useEffect, useRef, useState } from 'react';
import { getApi } from '@/api/currentApi';
import { ApiError, type ErrorCode } from '@/api/errors';
import { recipeToItems } from '@/core/recipe';
import { basketStore } from '@/core/stores/basket';
import { useT } from '@/i18n';
import { Button } from '@/ui/Button';
import { Dialog } from '@/ui/Dialog';
import { ErrorCard } from '@/ui/ErrorCard';
import { pendingImportStore } from './pendingImport';

type Phase = { kind: 'checking' } | { kind: 'confirm' } | { kind: 'failed'; code: ErrorCode; detail: unknown };

function addPending(): void {
  const recipe = pendingImportStore.getState().recipe;
  if (recipe !== null) basketStore.getState().add(recipeToItems(recipe, new Date().toISOString()));
  pendingImportStore.getState().set(null);
}

/**
 * Applies the pending recipe to the loaded model: `recipeCheck`, then `ok && !sha_mismatch` adds the items
 * (origin `import`), `sha_mismatch` asks first, anything else shows the errors and drops the recipe. Calls
 * `onSettled` once the recipe is added or dropped (at once when none is pending).
 */
export function RecipeApply({ onSettled }: { onSettled: () => void }) {
  const t = useT();
  const [phase, setPhase] = useState<Phase>({ kind: 'checking' });
  const started = useRef(false);

  useEffect(() => {
    if (started.current) return; // StrictMode runs effects twice; one check is enough
    started.current = true;
    const recipe = pendingImportStore.getState().recipe;
    if (recipe === null) {
      onSettled();
      return;
    }
    getApi()
      .recipeCheck({ recipe })
      .then((check) => {
        if (check.ok && !check.sha_mismatch) {
          addPending();
          onSettled();
        } else if (check.ok) {
          setPhase({ kind: 'confirm' });
        } else {
          setPhase({ kind: 'failed', code: 'invalid_input', detail: check.errors });
        }
      })
      .catch((error: unknown) => {
        setPhase(
          error instanceof ApiError
            ? { kind: 'failed', code: error.code, detail: error.detail }
            : { kind: 'failed', code: 'internal', detail: String(error) },
        );
      });
  }, [onSettled]);

  const drop = () => {
    pendingImportStore.getState().set(null);
    onSettled();
  };

  if (phase.kind === 'failed') {
    return (
      <div className="recipe-apply">
        <p>{t('recipe.checkFailed')}</p>
        <ErrorCard code={phase.code} detail={phase.detail} />
        <Button onClick={drop}>{t('recipe.continue')}</Button>
      </div>
    );
  }
  return (
    <>
      {phase.kind === 'checking' && <p className="recipe-apply">{t('recipe.checking')}</p>}
      <Dialog
        open={phase.kind === 'confirm'}
        title={t('recipe.sha.title')}
        onCancel={drop}
        actions={
          <>
            <Button
              variant="primary"
              onClick={() => {
                addPending();
                onSettled();
              }}
            >
              {t('recipe.sha.yes')}
            </Button>
            <Button onClick={drop}>{t('recipe.sha.no')}</Button>
          </>
        }
      >
        <p>{t('recipe.sha.confirm')}</p>
      </Dialog>
    </>
  );
}
