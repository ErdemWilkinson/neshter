import { useStore } from 'zustand';
import { navigate } from '@/app/router';
import { countSavable } from '@/core/basket';
import { basketStore } from '@/core/stores/basket';
import { saveFlowStore } from '@/features/saveflow/saveFlowStore';
import { plural, useT } from '@/i18n';
import { Button } from '@/ui/Button';

/**
 * After the save flow re-ran the comparison without the explore-only cuts. The second confirmation is this refreshed
 * screen plus an ordinary Save click on the Save screen: this button only navigates, there is no modal (F19).
 */
export function RefreshedBanner() {
  const t = useT();
  const flow = useStore(saveFlowStore, (s) => s.state);
  const items = useStore(basketStore, (s) => s.items);
  if (flow.kind !== 'reviewReady') return null;
  return (
    <div className="refreshed-banner glass" role="status">
      <p>{t('compare.refreshed', { n: flow.dropped })}</p>
      <Button variant="primary" onClick={() => navigate('save')}>
        {plural('basket.save', countSavable(items))}
      </Button>
    </div>
  );
}
