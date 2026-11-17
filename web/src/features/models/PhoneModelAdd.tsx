import { useState } from 'react';
import { getApi } from '@/api/currentApi';
import { asPhoneApi } from '@/api/device/deviceApi';
import { ApiError, type ErrorCode } from '@/api/errors';
import { useT } from '@/i18n';
import { Button } from '@/ui/Button';
import { ErrorCard } from '@/ui/ErrorCard';
import { formatBytes } from './format';

type State =
  | { kind: 'idle' }
  | { kind: 'copying'; done: number; total: number }
  | { kind: 'added'; name: string }
  | { kind: 'failed'; code: ErrorCode; detail: unknown };

/** Phone only: bring in a GGUF from the phone's storage (the system picker), then refresh the list. */
export function PhoneModelAdd({ onAdded }: { onAdded: () => void }) {
  const t = useT();
  const phone = asPhoneApi(getApi());
  const [state, setState] = useState<State>({ kind: 'idle' });
  if (phone === null) return null;

  const add = async () => {
    setState({ kind: 'copying', done: 0, total: -1 });
    try {
      const m = await phone.importModel((done, total) => setState({ kind: 'copying', done, total }));
      setState({ kind: 'added', name: m.name });
      onAdded();
    } catch (error) {
      if (error instanceof ApiError && error.code === 'cancelled') return setState({ kind: 'idle' });
      setState(error instanceof ApiError ? { kind: 'failed', code: error.code, detail: error.detail } : { kind: 'failed', code: 'internal', detail: String(error) });
    }
  };

  const busy = state.kind === 'copying';
  return (
    <div className="phone-model-add glass">
      <Button variant="primary" disabled={busy} onClick={() => void add()}>
        {t('models.add')}
      </Button>
      <p className="model-meta">{t('models.add.hint')}</p>
      {busy && (
        <p className="num" role="status">
          {state.done === 0
            ? t('models.add.checking')
            : t('models.add.copying', { done: formatBytes(state.done), total: state.total > 0 ? formatBytes(state.total) : '?' })}
        </p>
      )}
      {busy && <Button onClick={() => void phone.stop()}>{t('chat.stop')}</Button>}
      {state.kind === 'added' && <p role="status">{t('models.add.done', { name: state.name })}</p>}
      {state.kind === 'failed' && <ErrorCard code={state.code} detail={state.detail} />}
    </div>
  );
}
