import { useStore } from 'zustand';
import { isDeviceApp } from '@/api/device/bridge';
import { uiStore, type View } from '@/core/stores/ui';
import { useT, type MessageKey } from '@/i18n';
import { Segmented } from '@/ui/Segmented';

/** Registered views. `3d` is Plan 3 M2: the estimated connection graph of one answer. */
export const VIEWS: readonly { id: View; enabled: boolean }[] = [
  { id: 'hot', enabled: true },
  { id: '2d', enabled: true },
  { id: 'list', enabled: true },
  { id: '3d', enabled: true },
];

const LABELS: Record<View, MessageKey> = {
  hot: 'map.view.hot',
  '2d': 'map.view.2d',
  list: 'map.view.list',
  '3d': 'map.view.3d',
};

/** The phone has no connection graph (it needs gradients): `3d` is off there. */
const available = (v: { id: View; enabled: boolean }): boolean => v.enabled && !(v.id === '3d' && isDeviceApp());

export const isEnabledView = (view: View): boolean => VIEWS.some((v) => v.id === view && available(v));

/** The view to show: the chosen one, or the heatmap when it is not available (e.g. `3d` before Plan 3). */
export const effectiveView = (view: View): View => (isEnabledView(view) ? view : '2d');

export function ViewSwitch() {
  const t = useT();
  const view = useStore(uiStore, (s) => effectiveView(s.view));
  const options = VIEWS.filter(available).map((v) => ({ value: v.id, label: t(LABELS[v.id]) }));
  return (
    <Segmented label={t('map.view')} options={options} value={view} onChange={(v) => uiStore.getState().setView(v)} />
  );
}
