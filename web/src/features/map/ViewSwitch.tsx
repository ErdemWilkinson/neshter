import { useStore } from 'zustand';
import { uiStore, type View } from '@/core/stores/ui';
import { useT, type MessageKey } from '@/i18n';
import { Segmented } from '@/ui/Segmented';

/** Registered views. `3d` is Plan 3's slot: it exists here but is not rendered until it is enabled. */
export const VIEWS: readonly { id: View; enabled: boolean }[] = [
  { id: '2d', enabled: true },
  { id: 'list', enabled: true },
  { id: '3d', enabled: false },
];

const LABELS: Record<View, MessageKey> = {
  '2d': 'map.view.2d',
  list: 'map.view.list',
  '3d': 'map.view.3d',
};

export const isEnabledView = (view: View): boolean => VIEWS.some((v) => v.id === view && v.enabled);

/** The view to show: the chosen one, or the heatmap when it is not available (e.g. `3d` before Plan 3). */
export const effectiveView = (view: View): View => (isEnabledView(view) ? view : '2d');

export function ViewSwitch() {
  const t = useT();
  const view = useStore(uiStore, (s) => effectiveView(s.view));
  const options = VIEWS.filter((v) => v.enabled).map((v) => ({ value: v.id, label: t(LABELS[v.id]) }));
  return (
    <Segmented label={t('map.view')} options={options} value={view} onChange={(v) => uiStore.getState().setView(v)} />
  );
}
