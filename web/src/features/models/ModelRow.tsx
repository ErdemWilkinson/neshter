import { useId } from 'react';
import type { ModelEntry } from '@/api/schemas';
import { useT } from '@/i18n';
import { Button } from '@/ui/Button';
import { badgeLabel, formatArch, formatBytes, reasonLabel } from './format';

/** One model of the list. Everything from the server (name, source, license) is rendered as text. */
export function ModelRow({
  row,
  onOpen,
  onDelete,
}: {
  row: ModelEntry;
  onOpen: (row: ModelEntry) => void;
  /** Phone only: models the user brought in can be removed again. */
  onDelete?: ((row: ModelEntry) => void) | undefined;
}) {
  const t = useT();
  const nameId = useId();
  const reason = row.supported ? null : reasonLabel(row.unsupported_reason);
  return (
    <li className="model-row glass" aria-disabled={!row.supported || undefined}>
      <div className="model-row-main">
        <p className="model-name" id={nameId}>
          {row.name}
        </p>
        <p className="model-meta">
          {row.source} · {row.fmt.toLowerCase() === row.source.toLowerCase() ? '' : `${row.fmt} · `}
          <span className="num">{t('models.size', { size: formatBytes(row.size_bytes) })}</span>
        </p>
        <p className="model-meta">{row.arch === null ? t('models.arch.unknown') : formatArch(row.arch)}</p>
        <p className="model-meta">
          <span className="num">{t('models.memory', { size: formatBytes(row.memory_bytes) })}</span> ·{' '}
          {row.license ?? t('models.license.unknown')}
        </p>
        {reason !== null && <p className="model-reason">{reason}</p>}
      </div>
      <span className={`badge badge-fit-${row.badge}`}>{badgeLabel(row.badge)}</span>
      <Button variant="primary" disabled={!row.supported} aria-describedby={nameId} onClick={() => onOpen(row)}>
        {t('models.open')}
      </Button>
      {onDelete !== undefined && (
        <Button aria-describedby={nameId} onClick={() => onDelete(row)}>
          {t('models.delete')}
        </Button>
      )}
    </li>
  );
}
