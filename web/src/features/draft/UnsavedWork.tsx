import { useEffect, useState } from 'react';
import { getApi } from '@/api/currentApi';
import { ApiError, type ErrorCode } from '@/api/errors';
import type { DraftSummary } from '@/api/schemas';
import { plural, useT } from '@/i18n';
import { Button } from '@/ui/Button';
import { Dialog } from '@/ui/Dialog';
import { ErrorCard } from '@/ui/ErrorCard';
import { draftStatusStore } from './draftStatus';

/**
 * Home section: one row per server draft that still holds unsaved cuts (`unsaved_count > 0`, F4), hidden when none
 * does. Model names are rendered as text. `Taslakları sil` deletes every listed draft after a confirmation.
 */
export function UnsavedWork() {
  const t = useT();
  const [rows, setRows] = useState<DraftSummary[]>([]);
  const [confirming, setConfirming] = useState(false);
  const [failure, setFailure] = useState<{ code: ErrorCode; detail: unknown } | null>(null);

  useEffect(() => {
    let cancelled = false;
    getApi()
      .drafts()
      .then((all) => {
        if (!cancelled) setRows(all.filter((row) => row.unsaved_count > 0));
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  const deleteAll = async () => {
    setConfirming(false);
    const api = getApi();
    try {
      for (const row of rows) await api.deleteDraft(row.sha256);
      setRows([]);
      setFailure(null);
      draftStatusStore.getState().setHasDraft(false);
    } catch (error) {
      setFailure(error instanceof ApiError ? { code: error.code, detail: error.detail } : { code: 'internal', detail: String(error) });
    }
  };

  if (rows.length === 0 && failure === null) return null;
  return (
    <section className="unsaved-work">
      <ul>
        {rows.map((row) => (
          <li key={row.sha256}>{plural('unsaved.row', row.unsaved_count, { model_name: row.model_name })}</li>
        ))}
      </ul>
      <p>{t('unsaved.prompts')}</p>
      {failure !== null && <ErrorCard code={failure.code} detail={failure.detail} />}
      {rows.length > 0 && <Button onClick={() => setConfirming(true)}>{t('unsaved.deleteAll')}</Button>}
      <Dialog
        open={confirming}
        title={t('unsaved.confirmTitle')}
        onCancel={() => setConfirming(false)}
        actions={
          <>
            <Button variant="primary" onClick={() => void deleteAll()}>
              {t('unsaved.confirmYes')}
            </Button>
            <Button onClick={() => setConfirming(false)}>{t('leave.cancel')}</Button>
          </>
        }
      >
        <p>{t('unsaved.confirmBody')}</p>
      </Dialog>
    </section>
  );
}
