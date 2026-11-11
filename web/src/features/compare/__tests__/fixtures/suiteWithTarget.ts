import type { SuiteResult } from '@/api/schemas';

/**
 * Hand-made (the contract fixture `job_suite.json` is shape-only, F13): `code` is the target row with a clear drop,
 * `general` is an uncertain row (its interval crosses zero).
 */
export const suiteWithTarget: SuiteResult = {
  categories: [
    {
      category: 'code',
      n: 25,
      k_orig: 20,
      k_int: 5,
      a: 5,
      b: 15,
      c: 0,
      d: 5,
      broken_orig: 0,
      broken_int: 1,
      diff: { diff: -0.6, lo: -0.8, hi: -0.35, uncertain: false, p_value: 0.0001, p_adj: 0.0002 },
      orig_ci: [0.61, 0.91],
      int_ci: [0.09, 0.39],
    },
    {
      category: 'general',
      n: 25,
      k_orig: 2,
      k_int: 3,
      a: 1,
      b: 1,
      c: 2,
      d: 21,
      broken_orig: 2,
      broken_int: 3,
      diff: { diff: 0.04, lo: -0.1, hi: 0.15, uncertain: true, p_value: 0.5, p_adj: 0.5 },
      orig_ci: [0.02, 0.25],
      int_ci: [0.04, 0.3],
    },
  ],
  summary: {
    level: 'good',
    tr: 'Hedef davranış belirgin şekilde azaldı.',
    en: 'The target behavior dropped clearly.',
    mde_points: 18,
  },
  metrics: {
    n_prompts: 40,
    kl_random_note: null,
    kl: 0.0031,
    kl_std: 0.001,
    kl_random: 0.0042,
    kl_random_p: 0.25,
    kl_percentile: 80,
    direction_control: null,
    perplexity_orig: 12.5,
    perplexity_int: 12.9,
  },
  quick: true,
};
