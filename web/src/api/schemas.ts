// Response schemas, one per API route (named after the Api method). They mirror neshter/server/app.py and are
// checked against live shapes by tests/test_web_contract.py via src/api/__fixtures__. Loose objects: keys this
// version does not know pass through, so a newer server never breaks an older UI.
import { z } from '@/core/zod';
import { ERROR_CODES, type ErrorCode } from './errors';

const nullableNumber = z.number().nullable();
const ci = z.tuple([z.number(), z.number()]);
const errorCode = z.enum(ERROR_CODES);

// -- system / models --

export const system = z.looseObject({
  gpu_name: z.string().nullable(),
  vram_bytes: z.number(),
  ram_bytes: z.number(),
  default_out_dir: z.string(),
  llama_cpp: z.looseObject({ found: z.boolean(), path: z.string().nullable(), converter: z.boolean() }),
  ollama: z.boolean(),
  torch_version: z.string(),
  torch_cuda: z.string().nullable(),
  cuda_available: z.boolean(),
  native_bf16: z.boolean(),
  cpu_threads: z.number(),
});

export const archInfo = z.looseObject({
  family: z.string(),
  n_layers: z.number(),
  n_heads: z.number(),
  n_kv_heads: z.number(),
  head_dim: z.number(),
  d_model: z.number(),
  d_mlp: z.number(),
  n_params: z.number(),
});

export const modelEntry = z.looseObject({
  name: z.string(),
  source: z.string(),
  path: z.string(),
  fmt: z.string(),
  size_bytes: z.number(),
  memory_bytes: z.number(),
  badge: z.enum(['ok', 'slow', 'no']),
  license: z.string().nullable(),
  // Listing-time header read: no head_dim / d_model / n_params. Hostile or odd metadata gives null counts
  // (scan._arch_int), and one such model must not break the whole list.
  arch: z
    .looseObject({
      family: z.string(),
      n_layers: nullableNumber,
      n_heads: nullableNumber,
      n_kv_heads: nullableNumber,
      d_mlp: nullableNumber,
    })
    .nullable(),
  supported: z.boolean(),
  unsupported_reason: z.string().nullable(),
});
export const models = z.array(modelEntry);

export const concept = z.looseObject({
  id: z.string(),
  name_tr: z.string(),
  name_en: z.string(),
  safety_ablation: z.boolean(),
  suite_target: z.string().nullable(),
  positive: z.array(z.string()),
  negative: z.array(z.string()),
});
export const concepts = z.array(concept);

// -- session --

const sessionEmpty = z.looseObject({ loaded: z.literal(false) });
const sessionLoaded = z.looseObject({
  loaded: z.literal(true),
  name: z.string(),
  arch: archInfo,
  device: z.string(),
  dtype: z.string(),
  license: z.string().nullable(),
  memory_bytes: z.number(),
  device_note: z.string().nullable(),
  source_path: z.string(),
  source_sha256: z.string(),
  default_method: z.enum(['attribution', 'activation_diff', 'direction']),
  safety_dirs: z.number(),
  safety_heads: z.number(),
  measure_count: z.number().default(0),
});
/** `{loaded:false}` carries nothing else (no `measure_count`); narrow on `loaded`. */
export const session = z.discriminatedUnion('loaded', [sessionEmpty, sessionLoaded]);

// -- instant routes --

export const activity = z.looseObject({
  tokens: z.array(z.string()),
  power_raw: z.array(z.array(z.array(z.number()))), // [layer][head][token]
  power: z.array(z.array(z.number())).optional(), // [layer][head], only when token_index was sent
});
const lensToken = z.looseObject({ token_id: z.number(), text: z.string(), p: z.number() });
/** `POST /api/lens`: what the model would answer after every layer (logit lens). Row 0 is the embedding (`layer: -1`). */
export const lens = z.looseObject({
  tokens: z.array(z.string()),
  position: z.number(),
  final: z.looseObject({ token_id: z.number(), text: z.string() }),
  layers: z.array(
    z.looseObject({ layer: z.number(), top: z.array(lensToken), p_final: z.number(), rank_final: z.number() }),
  ),
});
/** `POST /api/attention`: `[head][query][key]`, each row sums to 1. */
export const attention = z.looseObject({
  tokens: z.array(z.string()),
  layer: z.number(),
  patterns: z.array(z.array(z.array(z.number()))),
});
export const graphCut = z.looseObject({
  edge_ids: z.array(z.string()),
  metric_before: z.number(),
  metric_after: z.number(),
  delta_metric: z.number(),
  delta_estimated: z.number(),
  p_before: z.number(),
  p_after: z.number(),
  kl: z.number(),
  top_before: z.array(lensToken),
  top_after: z.array(lensToken),
});
export const mlp = z.array(z.looseObject({ layer: z.number(), neuron: z.number(), value: z.number() }));
export const compare = z.looseObject({
  original: z.string(),
  modified: z.string(),
  mode: z.string(),
  target_hit: z.looseObject({ original: z.boolean(), modified: z.boolean() }).nullable(),
});
export const recipeCheck = z.looseObject({
  ok: z.boolean(),
  errors: z.array(z.string()),
  sha_mismatch: z.boolean(),
});
export const estimate = z.looseObject({
  bytes_needed: z.number(),
  free_bytes: z.number(),
  enough: z.boolean(),
  suggested_name: z.string(),
  safety_ablated: z.boolean(),
});

// -- jobs --

export const jobStart = z.looseObject({ job_id: z.string() });

export type JobState<R = unknown> = {
  id: string;
  kind: string;
  state: 'queued' | 'running' | 'done' | 'error' | 'cancelled';
  progress: number;
  message: string;
  result: R | null;
  error: string | null;
  error_code: ErrorCode | null;
  started_at: string | null;
};

const jobFields = {
  id: z.string(),
  kind: z.string(),
  state: z.enum(['queued', 'running', 'done', 'error', 'cancelled']),
  progress: z.number(),
  message: z.string(),
  error: z.string().nullable(),
  error_code: errorCode.nullable(),
  started_at: z.string().nullable(),
};

export function jobStateSchema<R>(result: z.ZodType<R>) {
  return z.looseObject({ ...jobFields, result: result.nullable() });
}
export const job = jobStateSchema(z.unknown());
export const cancelJob = job;
/** `GET /api/jobs`: states without their (possibly large) results. */
export const jobs = z.array(z.looseObject({ ...jobFields, created_at: z.string(), finished_at: z.string().nullable() }));

export const loadResult = z.looseObject({
  arch: archInfo,
  name: z.string(),
  license: z.string().nullable(),
  device: z.string(),
  device_note: z.string().nullable(),
  dtype: z.string(),
  source_sha256: z.string(),
});
export const sampleResult = z.looseObject({ path: z.string() });

export const effectResult = z.looseObject({
  kl: z.number(),
  kl_std: z.number(),
  kl_se: z.number(),
  n_prompts: z.number(),
  random_kls: z.array(z.number()),
  random_kl: nullableNumber,
  random_p05: nullableNumber,
  random_p95: nullableNumber,
  percentile: nullableNumber,
  n_drawn: z.number(),
  n_max: z.number(),
  k_exceed: z.number(),
  tau: z.number(),
  p_value: z.number(),
  p_kind: z.enum(['full', 'besag_clifford']),
  p_min: z.number(),
  significant: z.boolean(),
  stopped_early: z.boolean(),
  layer_matched: z.boolean(),
  seed: z.number(),
  session_measure_index: z.number(),
});

export const huntResult = z.looseObject({
  method: z.enum(['attribution', 'activation_diff', 'direction']),
  layer_scores: z.array(z.number()),
  heads: z.array(
    z.looseObject({
      layer: z.number(),
      head: z.number(),
      score: z.number(),
      measured: z.looseObject({ signal_drop: nullableNumber, neutral_kl: z.number() }).nullable(),
    }),
  ),
  direction: z.string().nullable(), // base64 little-endian float32
  direction_shape: z.array(z.number()).nullable(),
  direction_layer: z.number().nullable(),
  ref_layer: z.number(),
  val_accuracy: z.number(),
  val_ci: ci,
  val_n: z.number(),
  /** Model-level sweep (INLP idea): extra directions found after the earlier ones were removed; base64 float32. */
  extra_directions: z.array(z.string()).optional(),
  sweep: z
    .array(z.looseObject({ removed: z.number(), layer: z.number().nullable(), residual_accuracy: z.number(), n: z.number() }))
    .optional(),
});

export const graphNode = z.looseObject({
  id: z.string(),
  kind: z.enum(['token', 'head', 'neuron', 'output']),
  layer: z.number(),
  index: z.number(),
  score: z.number(),
});
export const graphEdge = z.looseObject({
  id: z.string(),
  src: z.string(),
  dst: z.string(),
  channel: z.enum(['q', 'v', 'resid', 'logit']),
  weight: z.number(),
  badge: z.string(),
});
/** Plan 3 §7 `POST /api/graph` result (single answer). Every weight is an estimate. */
export const graphResult = z.looseObject({
  graph_id: z.string(),
  estimated: z.literal(true),
  warnings: z.array(z.string()),
  metric: z.looseObject({ kind: z.string(), token_id: z.number(), text: z.string(), value: z.number() }),
  candidates: z.array(z.looseObject({ token_id: z.number(), text: z.string() })),
  tokens: z.array(z.looseObject({ pos: z.number(), token_id: z.number(), text: z.string() })),
  n_layers: z.number(),
  n_heads: z.number(),
  nodes: z.array(graphNode),
  edges: z.array(graphEdge),
  rivers: z.looseObject({ trunk: z.array(z.number()), norms: z.array(z.number()) }),
  pruning: z.looseObject({ kept_nodes: z.number(), kept_edges: z.number(), candidate_edges: z.number() }),
});

export const suiteResult = z.looseObject({
  categories: z.array(
    z.looseObject({
      category: z.string(),
      n: z.number(),
      k_orig: z.number(),
      k_int: z.number(),
      a: z.number(),
      b: z.number(),
      c: z.number(),
      d: z.number(),
      broken_orig: z.number(),
      broken_int: z.number(),
      diff: z.looseObject({
        diff: z.number(),
        lo: z.number(),
        hi: z.number(),
        uncertain: z.boolean(),
        p_value: nullableNumber,
        p_adj: nullableNumber,
      }),
      orig_ci: ci,
      int_ci: ci,
    }),
  ),
  summary: z
    .looseObject({ level: z.enum(['good', 'mixed', 'bad']), tr: z.string(), en: z.string(), mde_points: z.number() })
    .nullable(),
  metrics: z.looseObject({
    n_prompts: z.number(),
    kl_random_note: z.string().nullable(),
    kl: z.number(),
    kl_std: nullableNumber,
    kl_random: nullableNumber,
    kl_random_p: nullableNumber,
    kl_percentile: nullableNumber,
    direction_control: z.string().nullable(),
    perplexity_orig: z.number(),
    perplexity_int: z.number(),
  }),
  quick: z.boolean(),
});

export const exportResult = z.looseObject({
  path: z.string(),
  source_sha256: z.string(),
  source_untouched: z.boolean(),
  safety_ablated: z.boolean(),
  name: z.string(),
});

// -- server-side drafts and prefs --

export const draftSummary = z.looseObject({
  sha256: z.string(),
  model_name: z.string(),
  updated_at: z.string(),
  unsaved_count: z.number(),
});
export const drafts = z.array(draftSummary);
/** Ops are opaque to the server and to this schema: Task 6's codec decodes them (unknown types stay opaque). */
export const draft = z.looseObject({
  schema_version: z.literal(1),
  sha256: z.string(),
  model_name: z.string(),
  unsaved_count: z.number(),
  basket: z.array(z.record(z.string(), z.unknown())),
  shelf: z.array(z.record(z.string(), z.unknown())),
  saved_ops: z.array(z.string()),
  prompts: z.array(z.string()),
  updated_at: z.string(),
});
export const putDraft = draft;
export const deleteDraft = z.undefined();
export const prefs = z.looseObject({
  lang: z.enum(['tr', 'en']).nullable(),
  quality: z.enum(['high', 'balanced', 'low']),
  reduced_motion: z.enum(['system', 'on', 'off']),
  refusal_card_seen: z.boolean(),
});
export const putPrefs = prefs;
export const shutdown = z.looseObject({ ok: z.boolean() });

export const errorBody = z.looseObject({ detail: z.unknown(), error_code: errorCode });

export type SystemInfo = z.infer<typeof system>;
export type ArchInfo = z.infer<typeof archInfo>;
export type ModelEntry = z.infer<typeof modelEntry>;
export type Concept = z.infer<typeof concept>;
export type SessionInfo = z.infer<typeof session>;
export type LoadedSession = Extract<SessionInfo, { loaded: true }>;
export type ActivityResponse = z.infer<typeof activity>;
export type CompareResponse = z.infer<typeof compare>;
export type RecipeCheck = z.infer<typeof recipeCheck>;
export type ExportEstimate = z.infer<typeof estimate>;
export type LoadResult = z.infer<typeof loadResult>;
export type EffectResult = z.infer<typeof effectResult>;
export type HuntResult = z.infer<typeof huntResult>;
export type LensResponse = z.infer<typeof lens>;
export type AttentionResponse = z.infer<typeof attention>;
export type GraphResult = z.infer<typeof graphResult>;
export type GraphNode = z.infer<typeof graphNode>;
export type GraphEdge = z.infer<typeof graphEdge>;
export type GraphCut = z.infer<typeof graphCut>;
export type SuiteResult = z.infer<typeof suiteResult>;
export type ExportResult = z.infer<typeof exportResult>;
export type DraftSummary = z.infer<typeof draftSummary>;
export type DraftBody = z.infer<typeof draft>;
export type Prefs = z.infer<typeof prefs>;

/** Contract fixture name -> schema; `tests/test_web_contract.py` keeps the fixtures equal to live shapes. */
export const FIXTURE_SCHEMAS = {
  system,
  session_empty: session,
  session_loaded: session,
  models,
  concepts,
  activity,
  mlp,
  compare,
  recipe_check: recipeCheck,
  job_effect: jobStateSchema(effectResult),
  job_hunt: jobStateSchema(huntResult),
  job_suite: jobStateSchema(suiteResult),
  estimate,
  job_export: jobStateSchema(exportResult),
  jobs,
  drafts,
  draft,
  prefs,
  error_busy: errorBody,
} as const;
