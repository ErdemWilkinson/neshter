import type { ErrorCode } from '@/api/errors';
import { useT } from '@/i18n';
import { TechDetail } from './TechDetail';

function detailText(detail: unknown): string | null {
  if (detail === undefined || detail === null || detail === '') return null;
  if (typeof detail === 'string') return detail;
  try {
    return JSON.stringify(detail, null, 2);
  } catch {
    return String(detail);
  }
}

/** The message is chosen by `code`; the server's `detail` only ever appears inside the disclosure. */
export function ErrorCard({ code, detail }: { code: ErrorCode; detail?: unknown }) {
  const t = useT();
  const text = detailText(detail);
  return (
    <div className="error-card glass" role="alert">
      <p>{t(`errors.${code}`)}</p>
      {text !== null && <TechDetail>{text}</TechDetail>}
    </div>
  );
}
