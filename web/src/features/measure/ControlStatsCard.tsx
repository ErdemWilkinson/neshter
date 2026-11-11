import type { Measurement } from '@/core/measurement';
import { verdict } from '@/core/measurement';
import { useT } from '@/i18n';
import { TechDetail } from '@/ui/TechDetail';
import { formatStat } from './format';

/**
 * Spec §3: a plain-language verdict first (from the same random-control data), then the KL of the cut next to what
 * random head sets from the same layers do (never a bare number) as smaller, secondary detail.
 */
export function ControlStatsCard({ m }: { m: Measurement }) {
  const t = useT();
  const c = m.control;
  const kind = verdict(m);
  const summary =
    kind === 'significant'
      ? t('measure.summary.significant')
      : kind === 'no_control'
        ? t('measure.summary.noControl')
        : t('measure.summary.insignificant');
  return (
    <div className="control-card glass">
      <div className="control-head">
        <h3>{t('measure.title')}</h3>
        <span className="badge badge-measured">{t('measure.badge')}</span>
      </div>
      <p className={`control-summary control-summary-${kind === 'significant' ? 'yes' : 'no'}`}>{summary}</p>
      <div className="control-detail">
        <p className="num">
          {m.kl_se === null
            ? t('measure.kl.noSe', { kl: formatStat(m.kl) })
            : t('measure.kl', { kl: formatStat(m.kl), se: formatStat(m.kl_se) })}
          {' · '}
          {t('measure.prompts', { n: m.n_prompts })}
        </p>
        {c !== null && c.band !== null && (
          <p className="num">{t('measure.band', { lo: formatStat(c.band[0]), hi: formatStat(c.band[1]) })}</p>
        )}
        {c !== null && kind === 'significant' && (
          <p>{t('measure.verdict.significant', { p: formatStat(c.p_value), n: c.n_drawn })}</p>
        )}
        {c !== null && kind === 'not_significant' && (
          <p>{t('measure.verdict.notSignificant', { p: formatStat(c.p_value), tau: formatStat(c.tau), n: c.n_drawn })}</p>
        )}
        {c !== null && kind === 'futility' && (
          <p>{t('measure.verdict.futility', { tau: formatStat(c.tau), n: c.n_drawn, k: c.k_exceed })}</p>
        )}
      </div>
      {m.session_index >= 2 && <p className="control-session">{t('measure.session', { k: m.session_index })}</p>}
      {c !== null && (
        <TechDetail>{t('measure.tech', { seed: String(m.seed), kind: c.p_kind, k: c.k_exceed, n: c.n_drawn })}</TechDetail>
      )}
    </div>
  );
}
