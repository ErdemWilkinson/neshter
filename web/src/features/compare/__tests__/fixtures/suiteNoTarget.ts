import type { SuiteResult } from '@/api/schemas';
import { suiteWithTarget } from './suiteWithTarget';

/** A concept without a test category (tone): no summary, no target row marked, only the general numbers. */
export const suiteNoTarget: SuiteResult = {
  ...suiteWithTarget,
  summary: null,
  metrics: { ...suiteWithTarget.metrics, kl_random: null, kl_random_p: null, kl_percentile: null, kl_random_note: 'Kontrol çekişi yapılmadı.' },
};
