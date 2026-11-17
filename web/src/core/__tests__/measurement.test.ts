import jobEffectFixture from '@/api/__fixtures__/job_effect.json';
import { effectResult, jobStateSchema } from '@/api/schemas';
import { fromEffectResult, verdict } from '../measurement';
import { futilityEffect, notSignificantEffect, significantEffect } from './fixtures/futilityEffect';
import { NOW, must } from './fixtures/items';

test('fromEffectResult maps futility and significance', () => {
  const futile = fromEffectResult(futilityEffect, 'fp', NOW);
  expect(futile).toMatchObject({
    kind: 'set_effect',
    value: futilityEffect.kl,
    se: futilityEffect.kl_se,
    kl: futilityEffect.kl,
    kl_se: futilityEffect.kl_se,
    uncertain: true,
    n_prompts: 2,
    seed: 7,
    session_index: 3,
    ops_fp: 'fp',
    measured_at: NOW,
    control: { n_drawn: 12, n_max: 39, k_exceed: 6, p_kind: 'besag_clifford', significant: false, stopped_early: true, tau: 0.05, band: [0.0011, 0.0038], percentile: 31.5 },
  });
  expect(verdict(futile)).toBe('futility');

  const full = fromEffectResult(notSignificantEffect, 'fp', NOW);
  expect(full.control?.p_kind).toBe('full');
  expect(verdict(full)).toBe('not_significant');

  const sig = fromEffectResult(significantEffect, 'fp', NOW);
  expect(sig.uncertain).toBe(false);
  expect(sig.control?.band).toBeNull(); // no random band reported
  expect(verdict(sig)).toBe('significant');
});

test('verdict without a control is no_control', () => {
  expect(verdict({ ...fromEffectResult(significantEffect, 'fp', NOW), control: null })).toBe('no_control');
});

test('fromEffectResult takes the shape of a real job_effect fixture', () => {
  const job = jobStateSchema(effectResult).parse(jobEffectFixture);
  const m = fromEffectResult(must(job.result), 'fp', NOW);
  expect(verdict(m)).toBe('significant');
  expect(m.session_index).toBe(1);
  expect(m.control?.band).toHaveLength(2);
});

test('fromEffectResult refuses a payload that is not an effect result (job results are not validated upstream)', () => {
  expect(() => fromEffectResult({ kl: 'x' } as never, 'fp', NOW)).toThrow();
  expect(() => fromEffectResult({ ...futilityEffect, p_kind: 'exact' } as never, 'fp', NOW)).toThrow();
});

test('fromEffectResult keeps the probe answer, and an effect without one has none', () => {
  const probe = {
    answer: { token_id: 5, text: ' dünya' },
    p_before: 0.41,
    p_after: 0.2,
    top_before: [{ token_id: 5, text: ' dünya', p: 0.41 }],
    top_after: [{ token_id: 7, text: ' ev', p: 0.3 }],
  };
  expect(fromEffectResult({ ...notSignificantEffect, probe }, 'fp', NOW).probe).toEqual(probe);
  expect(fromEffectResult(notSignificantEffect, 'fp', NOW).probe).toBeNull();
});
