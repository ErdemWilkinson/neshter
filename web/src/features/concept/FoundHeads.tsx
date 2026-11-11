import { useMemo, useState } from 'react';
import type { HuntResult } from '@/api/schemas';
import { navigate } from '@/app/router';
import { headId } from '@/core/ids';
import { selectionStore } from '@/core/stores/selection';
import { formatStat } from '@/features/measure/format';
import { useT } from '@/i18n';
import { Button } from '@/ui/Button';

type FoundHead = HuntResult['heads'][number];

const PAGE = 8;

function Measured({ measured }: { measured: FoundHead['measured'] }) {
  const t = useT();
  if (measured === null) return <>{t('concept.heads.unmeasured')}</>;
  const kl = formatStat(measured.neutral_kl);
  // `signal_drop` is null for a head at or after the reference layer: it cannot reach the signal (not "no effect").
  return measured.signal_drop === null ? (
    <>{t('concept.heads.measuredNoSignal', { kl })}</>
  ) : (
    <>{t('concept.heads.measuredValue', { drop: formatStat(measured.signal_drop), kl })}</>
  );
}

/**
 * The best heads by estimated score, eight at a time. The score is an estimate; what was measured (when it was) sits
 * next to it. Heads are cut only from the measured map, so the way on is `Haritada göster ve ölç`.
 */
export function FoundHeads({ heads }: { heads: readonly FoundHead[] }) {
  const t = useT();
  const [shown, setShown] = useState(PAGE);
  const sorted = useMemo(() => [...heads].sort((a, b) => b.score - a.score), [heads]);
  const visible = sorted.slice(0, shown);
  if (sorted.length === 0) return null;

  const showOnMap = () => {
    selectionStore.getState().set(visible.map((found) => headId(found.layer, found.head)));
    navigate('explore');
  };

  return (
    <section className="found-heads" aria-labelledby="found-heads-title">
      <h3 id="found-heads-title">{t('concept.heads.title')}</h3>
      <table className="found-heads-table">
        <caption className="sr-only">{t('concept.heads.caption')}</caption>
        <thead>
          <tr>
            <th scope="col">{t('map.table.layer')}</th>
            <th scope="col">{t('map.table.head')}</th>
            <th scope="col">{t('concept.heads.score')}</th>
            <th scope="col">{t('concept.heads.measured')}</th>
          </tr>
        </thead>
        <tbody>
          {visible.map((found) => (
            <tr key={`${found.layer}:${found.head}`}>
              <td className="num">{found.layer}</td>
              <td className="num">{found.head}</td>
              <td className="num">{found.score.toFixed(2)}</td>
              <td className="num">
                <Measured measured={found.measured} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="found-heads-actions">
        {shown < sorted.length && (
          <Button onClick={() => setShown((n) => n + PAGE)}>{t('concept.heads.more')}</Button>
        )}
        <Button variant="primary" onClick={showOnMap}>
          {t('concept.heads.show')}
        </Button>
      </div>
    </section>
  );
}
