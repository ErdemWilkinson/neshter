import { useStore } from 'zustand';
import { useT } from '@/i18n';
import { Segmented } from '@/ui/Segmented';
import { exploreStore } from './exploreStore';

/** `Katman içi` (default) or `Tüm model`; the switch only re-normalises the numbers already on screen. */
export function ScaleToggle() {
  const t = useT();
  const scale = useStore(exploreStore, (s) => s.scale);
  return (
    <Segmented
      label={t('map.scale')}
      options={[
        { value: 'layer', label: t('map.scale.layer') },
        { value: 'global', label: t('map.scale.global') },
      ]}
      value={scale}
      onChange={(next) => exploreStore.getState().setScale(next)}
    />
  );
}
