import { useT } from '@/i18n';
import { Button } from '@/ui/Button';

/** The empty-state offer; the download job and the load that follows it run in `LoadScreen`. */
export function SampleDownload({ onStart }: { onStart: () => void }) {
  const t = useT();
  return (
    <Button variant="primary" onClick={onStart}>
      {t('models.sample')}
    </Button>
  );
}
