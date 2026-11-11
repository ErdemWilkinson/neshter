import { useStore } from 'zustand';
import { ResultCard } from '@/features/save/ResultCard';
import { SaveForm } from '@/features/save/SaveForm';
import { SaveFlowDialogs } from '@/features/saveflow/SaveFlowDialogs';
import { saveFlowStore } from '@/features/saveflow/saveFlowStore';
import { useT } from '@/i18n';
import { Button } from '@/ui/Button';

/** Spec section 2 step 4: the last step. Arriving here never starts an export; only the Save button does (F19). */
export function Save() {
  const t = useT();
  const state = useStore(saveFlowStore, (s) => s.state);
  return (
    <section className="save-screen">
      <h1>{t('app.step.save')}</h1>
      {state.kind === 'done' ? (
        <>
          <ResultCard result={state.result} />
          <Button onClick={() => saveFlowStore.getState().dispatch({ type: 'DISMISS' })}>{t('save.again')}</Button>
        </>
      ) : (
        <SaveForm />
      )}
      <SaveFlowDialogs />
    </section>
  );
}
