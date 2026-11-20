import { useT } from '@/i18n';

type DotKind = 'kept' | 'lost' | 'gained' | 'none';

/** Which of the 100 dots are what: kept (still there), lost (taken by the cut), gained (added by it), none. */
export function dotKinds(before: number, after: number): DotKind[] {
  const b = Math.round(Math.min(1, Math.max(0, before)) * 100);
  const a = Math.round(Math.min(1, Math.max(0, after)) * 100);
  const kept = Math.min(a, b);
  return Array.from({ length: 100 }, (_, i): DotKind => (i < kept ? 'kept' : i < b ? 'lost' : i < a ? 'gained' : 'none'));
}

/**
 * The cut report as a unit chart: 100 dots, one per percent of the answer's probability. Beside it the old value
 * struck through and the new one large, what each dot colour means, what was cut and the model's new top guess.
 */
export function HundredDots({
  before,
  after,
  sentence,
  cuts,
  guess,
  changed,
}: {
  before: number;
  after: number;
  /** The plain sentence ("... probability: 41% → 20%"), shown under the headline and read by screen readers. */
  sentence: string;
  cuts: readonly string[];
  guess: string;
  changed: boolean;
}) {
  const t = useT();
  const kinds = dotKinds(before, after);
  const count = (k: DotKind) => kinds.filter((x) => x === k).length;
  const b = Math.round(before * 100);
  const a = Math.round(after * 100);
  return (
    <div className="hundred">
      <div className="hundred-grid" aria-hidden="true">
        {kinds.map((k, i) => (
          <span key={i} className={`hundred-dot is-${k}`} />
        ))}
      </div>
      <div className="hundred-side">
        <p className="hundred-head num" aria-hidden="true">
          <s>{t('jobs.percent', { pct: b })}</s> <b>{t('jobs.percent', { pct: a })}</b>
        </p>
        <p className="hundred-sentence">{sentence}</p>
        <ul className="hundred-key">
          <li>
            <span className="hundred-dot is-kept" aria-hidden="true" /> {t('graph.dots.kept', { n: count('kept') })}
          </li>
          {count('lost') > 0 && (
            <li className="is-lost">
              <span className="hundred-dot is-lost" aria-hidden="true" /> {t('graph.dots.lost', { n: count('lost') })}
            </li>
          )}
          {count('gained') > 0 && (
            <li>
              <span className="hundred-dot is-gained" aria-hidden="true" /> {t('graph.dots.gained', { n: count('gained') })}
            </li>
          )}
          <li>
            <span className="hundred-dot is-none" aria-hidden="true" /> {t('graph.dots.none', { n: count('none') })}
          </li>
        </ul>
        <p>
          <span className="hundred-label">{t('graph.dots.cut')}</span> <span className="num hundred-cuts">{cuts.join(' + ')}</span>
        </p>
        <p>
          <span className="hundred-label">{t('graph.dots.guess')}</span> <code>{guess}</code>{' '}
          <span className={changed ? 'hundred-changed' : undefined}>{changed ? t('graph.dots.changed') : t('graph.dots.same')}</span>
        </p>
      </div>
    </div>
  );
}
