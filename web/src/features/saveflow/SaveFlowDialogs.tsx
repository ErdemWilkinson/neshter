import { DiscardDialog } from './DiscardDialog';
import { NothingToSaveDialog } from './NothingToSaveDialog';
import { SaveFlowNotice } from './SaveFlowNotice';
import { useSaveFlow } from './useSaveFlow';

/**
 * The save flow's dialogs and its failure notice, driven by `useSaveFlow`. Mounted by the screens where a flow can be
 * in progress (Compare, Save); a state that needs no dialog renders nothing.
 */
export function SaveFlowDialogs() {
  const flow = useSaveFlow();
  const exploreIds = flow.state.kind === 'needsDiscard' ? flow.state.exploreIds : [];
  return (
    <>
      {flow.error !== null && <SaveFlowNotice error={flow.error} />}
      <DiscardDialog
        open={flow.state.kind === 'needsDiscard'}
        exploreIds={exploreIds}
        onConfirm={flow.confirmDiscard}
        onCancel={flow.cancel}
      />
      <NothingToSaveDialog open={flow.state.kind === 'nothingToSave'} onClose={flow.dismiss} />
    </>
  );
}
