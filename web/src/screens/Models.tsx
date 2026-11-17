import { useEffect, useState } from 'react';
import { isDeviceApp } from '@/api/device/bridge';
import { getApi } from '@/api/currentApi';
import { asPhoneApi } from '@/api/device/deviceApi';
import { ApiError, type ErrorCode } from '@/api/errors';
import type { ModelEntry } from '@/api/schemas';
import { LoadScreen, type LoadPlan } from '@/features/models/LoadScreen';
import { ModelRow } from '@/features/models/ModelRow';
import { PathField } from '@/features/models/PathField';
import { PhoneModelAdd } from '@/features/models/PhoneModelAdd';
import { SampleDownload } from '@/features/models/SampleDownload';
import { useLeaveGuard } from '@/features/leave/useLeaveGuard';
import { useT } from '@/i18n';
import { ErrorCard } from '@/ui/ErrorCard';

type Listing = { status: 'loading' } | { status: 'ready'; rows: ModelEntry[] } | { status: 'failed'; code: ErrorCode; detail: unknown };

export function Models() {
  const t = useT();
  const [listing, setListing] = useState<Listing>({ status: 'loading' });
  const [plan, setPlan] = useState<LoadPlan | null>(null);
  const guard = useLeaveGuard({ beforeUnload: false }); // App arms the tab-close confirmation

  const [version, setVersion] = useState(0); // bumping it lists the models again (one was added or deleted)
  const refresh = () => setVersion((v) => v + 1);
  const phone = asPhoneApi(getApi());

  useEffect(() => {
    let cancelled = false;
    getApi()
      .models()
      .then((rows) => {
        if (!cancelled) setListing({ status: 'ready', rows });
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        setListing(
          error instanceof ApiError
            ? { status: 'failed', code: error.code, detail: error.detail }
            : { status: 'failed', code: 'internal', detail: String(error) },
        );
      });
    return () => {
      cancelled = true;
    };
  }, [version]);

  const remove = (row: ModelEntry) => {
    if (phone === null || !window.confirm(t('models.delete.confirm', { name: row.name }))) return;
    void phone
      .deleteModel(row.path.slice('device:'.length))
      .then(refresh)
      .catch((error: unknown) =>
        setListing(error instanceof ApiError ? { status: 'failed', code: error.code, detail: error.detail } : { status: 'failed', code: 'internal', detail: String(error) }),
      );
  };

  // Switching models drops the cuts of the model on screen, so unsaved work asks first (autosave keeps the draft).
  const begin = (next: LoadPlan) => guard.request('switch_model', () => setPlan(next));

  if (plan !== null) return <LoadScreen plan={plan} onBack={() => setPlan(null)} />;

  return (
    <section className="models-screen">
      <h1>{t('app.step.models')}</h1>
      {phone !== null && <PhoneModelAdd onAdded={refresh} />}
      {listing.status === 'loading' && <p>{t('models.searching')}</p>}
      {listing.status === 'failed' && <ErrorCard code={listing.code} detail={listing.detail} />}
      {listing.status === 'ready' && listing.rows.length === 0 && (
        <div className="models-empty">
          <p>{t('models.empty')}</p>
          <p>{t('models.empty.hint')}</p>
          {!isDeviceApp() && <SampleDownload onStart={() => begin({ kind: 'sample' })} />}
        </div>
      )}
      {listing.status === 'ready' && listing.rows.length > 0 && (
        <ul className="model-list" aria-label={t('models.list')}>
          {listing.rows.map((row) => (
            <ModelRow
              key={row.path}
              row={row}
              onOpen={(picked) => begin({ kind: 'load', request: { path: picked.path, name: picked.name, dtype: 'auto' } })}
              onDelete={phone !== null && row.fmt === 'gguf' ? remove : undefined}
            />
          ))}
        </ul>
      )}
      {!isDeviceApp() && <PathField onOpen={(path) => begin({ kind: 'load', request: { path, dtype: 'auto' } })} />}
      {guard.dialog}
    </section>
  );
}
