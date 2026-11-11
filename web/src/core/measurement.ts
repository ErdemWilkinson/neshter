import { effectResult, type EffectResult } from '@/api/schemas';

/** The random-control outcome of a measurement (same fields as the server's effect result). */
export interface ControlStats {
  n_drawn: number;
  n_max: number;
  k_exceed: number;
  p_value: number;
  p_kind: 'full' | 'besag_clifford';
  p_min: number;
  tau: number;
  significant: boolean;
  stopped_early: boolean;
  band: [number, number] | null;
  percentile: number | null;
}

export interface Measurement {
  kind: 'set_effect' | 'edge' | 'steer';
  value: number;
  se: number | null;
  kl: number;
  kl_se: number | null;
  control: ControlStats | null;
  uncertain: boolean;
  n_prompts: number;
  seed: number;
  session_index: number;
  ops_fp: string;
  measured_at: string;
}

export type Verdict = 'significant' | 'not_significant' | 'futility' | 'no_control';

/**
 * Job results are typed but not runtime-validated by the job runner, so this is where an effect result is checked:
 * a malformed payload throws a ZodError instead of becoming a plausible-looking measurement.
 */
export function fromEffectResult(res: EffectResult, opsFp: string, now: string): Measurement {
  const r = effectResult.parse(res);
  return {
    kind: 'set_effect',
    value: r.kl,
    se: r.kl_se,
    kl: r.kl,
    kl_se: r.kl_se,
    control: {
      n_drawn: r.n_drawn,
      n_max: r.n_max,
      k_exceed: r.k_exceed,
      p_value: r.p_value,
      p_kind: r.p_kind,
      p_min: r.p_min,
      tau: r.tau,
      significant: r.significant,
      stopped_early: r.stopped_early,
      band: r.random_p05 !== null && r.random_p95 !== null ? [r.random_p05, r.random_p95] : null,
      percentile: r.percentile,
    },
    uncertain: !r.significant,
    n_prompts: r.n_prompts,
    seed: r.seed,
    session_index: r.session_measure_index,
    ops_fp: opsFp,
    measured_at: now,
  };
}

/** `futility`: the permutation test stopped early because the decision could no longer change (never significant). */
export function verdict(m: Measurement): Verdict {
  const c = m.control;
  if (c === null) return 'no_control';
  if (c.significant) return 'significant';
  return c.stopped_early ? 'futility' : 'not_significant';
}
