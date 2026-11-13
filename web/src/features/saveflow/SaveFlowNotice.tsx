import { useT } from '@/i18n';
import { ErrorCard } from '@/ui/ErrorCard';
import type { SaveFlowFailure } from './saveFlowStore';

/** The re-run failed or was refused (busy model, failed job): the basket is as it was; the reason is by code. */
export function SaveFlowNotice({ error }: { error: SaveFlowFailure }) {
  const t = useT();
  return (
    <div>
      <p>{t(error.stage === 'export' ? 'save.failed' : 'saveflow.failed')}</p>
      <ErrorCard code={error.code} detail={error.detail} />
    </div>
  );
}
