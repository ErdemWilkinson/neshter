import type { SuiteResult } from '@/api/schemas';
import { useT, type MessageKey } from '@/i18n';
import { points, rateWithCi } from './format';

const KNOWN = new Set(['code', 'general', 'math', 'turkish', 'harmful_refusal']);

/** `concept.category.<id>` for the five suite categories, the raw id (as text) for anything else. */
const categoryKey = (id: string): MessageKey | null => (KNOWN.has(id) ? (`concept.category.${id}` as MessageKey) : null);

/** Per-category counts with Wilson intervals and the paired difference; the target row is marked `(hedef)`. */
export function SuiteTable({ result, target }: { result: SuiteResult; target: string | null }) {
  const t = useT();
  const { summary } = result;
  return (
    <div className="suite-table-wrap">
      <table className="suite-table">
        <caption>{t('compare.table.caption')}</caption>
        <thead>
          <tr>
            <th scope="col">{t('compare.table.category')}</th>
            <th scope="col">{t('compare.table.n')}</th>
            <th scope="col">{t('compare.table.orig')}</th>
            <th scope="col">{t('compare.table.int')}</th>
            <th scope="col">{t('compare.table.diff')}</th>
            <th scope="col">{t('compare.table.broken')}</th>
          </tr>
        </thead>
        <tbody>
          {result.categories.map((row) => {
            const key = categoryKey(row.category);
            const isTarget = row.category === target;
            return (
              <tr key={row.category} className={isTarget ? 'suite-target' : undefined}>
                <th scope="row">
                  {key === null ? row.category : t(key)}
                  {isTarget && ` ${t('compare.table.target')}`}
                </th>
                <td className="num">{row.n}</td>
                <td className="num">{rateWithCi(row.k_orig, row.n, row.orig_ci)}</td>
                <td className="num">{rateWithCi(row.k_int, row.n, row.int_ci)}</td>
                <td className="num">
                  {t('compare.table.points', { d: points(row.diff.diff) })} [{points(row.diff.lo)}; {points(row.diff.hi)}]
                  {row.diff.uncertain && <span className="badge badge-uncertain"> {t('compare.table.uncertain')}</span>}
                </td>
                <td className="num">
                  {row.broken_orig} / {row.broken_int}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {summary !== null && <p className="suite-mde num">{t('compare.table.mde', { mde: summary.mde_points })}</p>}
    </div>
  );
}
