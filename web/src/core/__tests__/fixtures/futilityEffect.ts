// Hand-made effect results (the generated job_effect fixture is shape-only: its values are one tiny-model run).
import type { EffectResult } from '@/api/schemas';

const base: EffectResult = {
  kl: 0.0021,
  kl_std: 0.0004,
  kl_se: 0.0003,
  n_prompts: 2,
  random_kls: [0.0019, 0.0024, 0.0031],
  random_kl: 0.0025,
  random_p05: 0.0011,
  random_p95: 0.0038,
  percentile: 31.5,
  n_drawn: 39,
  n_max: 39,
  k_exceed: 20,
  tau: 0.05,
  p_value: 0.525,
  p_kind: 'full',
  p_min: 0.025,
  significant: false,
  stopped_early: false,
  layer_matched: true,
  seed: 7,
  session_measure_index: 3,
};

/** The permutation test stopped after 12 of 39 draws because the decision could no longer change. */
export const futilityEffect: EffectResult = {
  ...base,
  n_drawn: 12,
  k_exceed: 6,
  p_value: 0.5,
  p_kind: 'besag_clifford',
  stopped_early: true,
};

/** A full run that is not significant. */
export const notSignificantEffect: EffectResult = base;

/** A full run that is significant, with no random band reported. */
export const significantEffect: EffectResult = {
  ...base,
  k_exceed: 0,
  p_value: 0.025,
  significant: true,
  percentile: 100,
  random_p05: null,
  random_p95: null,
};
