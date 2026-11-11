import { useT } from '@/i18n';
import { Button } from '@/ui/Button';

/** Shown once, the first time a safety concept is chosen: the saved model will carry the `safety-ablated` label. */
export function RefusalCard({ onAcknowledge }: { onAcknowledge: () => void }) {
  const t = useT();
  return (
    <aside className="refusal-card glass" aria-labelledby="refusal-title">
      <h2 id="refusal-title">{t('concept.refusal.title')}</h2>
      <p>{t('concept.refusal.body')}</p>
      <Button variant="primary" onClick={onAcknowledge}>
        {t('concept.refusal.ok')}
      </Button>
    </aside>
  );
}
