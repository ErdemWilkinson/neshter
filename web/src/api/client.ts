import type { z } from '@/core/zod';
import { ApiError, BusyError, apiErrorFromResponse, isAbortError } from './errors';
import * as schemas from './schemas';
import type { DraftBody, JobState, Prefs } from './schemas';
import { streamSse } from './sse';

/** A recipe op as sent to the server: `{type:'head',...}` / `{type:'direction',...}` (Task 6 types them). */
export type JsonOp = { type: string; [key: string]: unknown };
type Method = 'auto' | 'attribution' | 'activation_diff' | 'direction';
type ExportFormat = 'safetensors' | 'gguf_q4_k_m' | 'gguf_q8_0';

export interface LoadRequest {
  path: string;
  name?: string;
  dtype?: 'auto' | 'bfloat16' | 'float32';
}
export interface ActivityRequest {
  prompt: string;
  token_index?: number | null;
  scale?: 'layer' | 'global';
}
export interface MlpRequest {
  prompt: string;
  token_index: number;
}
export interface CompareRequest {
  prompt: string;
  recipe_ops: JsonOp[];
  max_new_tokens?: number;
  concept_id?: string | null;
}
export interface EffectRequest {
  heads: [number, number][];
  prompts?: string[];
  /** 19..999; the server caps it at 199 on CPU (422 `invalid_input`). */
  n_max?: number;
  seed?: number;
}
export interface HuntRequest {
  concept_id?: string;
  concept?: { id: string; name_tr: string; name_en: string; positive: string[]; negative: string[]; [key: string]: unknown };
  method?: Exclude<Method, 'auto'> | null;
  seed?: number;
  /** >1 with `direction`: sweep the model for leaked directions after removing the found ones. */
  sweeps?: number;
}
export interface GraphRequest {
  prompt: string;
  target_token_id?: number | null;
  budget?: { heads_per_layer?: number; neurons_per_layer?: number; max_nodes?: number; max_edges?: number };
}
export interface SuiteRequest {
  recipe_ops: JsonOp[];
  target?: string | null;
  max_new_tokens?: number;
  quick?: boolean;
}
export interface EstimateRequest {
  format: ExportFormat;
  recipe_ops?: JsonOp[];
  has_directions?: boolean;
  out_dir?: string;
  concept?: string | null;
}
export interface ExportRequest {
  recipe_ops: JsonOp[];
  name?: string | null;
  format: ExportFormat;
  ollama: boolean;
  out_dir: string;
  concept?: string | null;
}

export interface StreamJobOptions {
  signal: AbortSignal;
  onEvent: (event: string, data: unknown) => void;
}

export interface Api {
  system(): Promise<schemas.SystemInfo>;
  models(): Promise<schemas.ModelEntry[]>;
  concepts(): Promise<schemas.Concept[]>;
  session(): Promise<schemas.SessionInfo>;
  loadModel(body: LoadRequest): Promise<{ job_id: string }>;
  sample(): Promise<{ job_id: string }>;
  activity(body: ActivityRequest): Promise<schemas.ActivityResponse>;
  mlp(body: MlpRequest): Promise<z.infer<typeof schemas.mlp>>;
  lens(body: { prompt: string; position?: number }): Promise<schemas.LensResponse>;
  attention(body: { prompt: string; layer: number }): Promise<schemas.AttentionResponse>;
  graph(body: GraphRequest): Promise<{ job_id: string }>;
  graphCut(graphId: string, edgeIds: string[]): Promise<schemas.GraphCut>;
  compare(body: CompareRequest): Promise<schemas.CompareResponse>;
  recipeCheck(body: { recipe: unknown }): Promise<schemas.RecipeCheck>;
  effect(body: EffectRequest): Promise<{ job_id: string }>;
  hunt(body: HuntRequest): Promise<{ job_id: string }>;
  suite(body: SuiteRequest): Promise<{ job_id: string }>;
  estimate(body: EstimateRequest): Promise<schemas.ExportEstimate>;
  exportModel(body: ExportRequest): Promise<{ job_id: string }>;
  jobs(): Promise<z.infer<typeof schemas.jobs>>;
  job<R = unknown>(id: string, signal?: AbortSignal): Promise<JobState<R>>;
  cancelJob<R = unknown>(id: string): Promise<JobState<R>>;
  drafts(): Promise<schemas.DraftSummary[]>;
  draft(sha: string): Promise<DraftBody>;
  putDraft(sha: string, body: Omit<DraftBody, 'updated_at'>): Promise<DraftBody>;
  deleteDraft(sha: string): Promise<void>;
  prefs(): Promise<Prefs>;
  putPrefs(prefs: Prefs): Promise<Prefs>;
  shutdown(): Promise<{ ok: boolean }>;
  /** `GET /api/jobs/{id}/events` through the same token and stale/busy handling; see `attachJob`. */
  streamJob(id: string, opts: StreamJobOptions): Promise<'ended' | 'aborted'>;
}

export interface ApiOptions {
  token: string;
  fetchFn?: typeof fetch;
  /** The token no longer fits (server restarted): called once per api instance. */
  onStale: () => void;
  /** The model is held by a job (409 `busy`). */
  onBusy?: () => void;
}

export function createApi(opts: ApiOptions): Api {
  const fetchFn: typeof fetch = opts.fetchFn ?? ((input, init) => fetch(input, init));
  let staleReported = false;

  function observe(error: ApiError): ApiError {
    if (error.code === 'forbidden' && !staleReported) {
      staleReported = true;
      opts.onStale();
    } else if (error instanceof BusyError) {
      opts.onBusy?.();
    }
    return error;
  }

  async function guarded<T>(run: () => Promise<T>): Promise<T> {
    try {
      return await run();
    } catch (error) {
      if (error instanceof ApiError) observe(error);
      throw error;
    }
  }

  async function request<S extends z.ZodType>(
    method: 'GET' | 'POST' | 'PUT' | 'DELETE',
    path: string,
    schema: S,
    body?: unknown,
    signal?: AbortSignal,
  ): Promise<z.infer<S>> {
    return guarded(async () => {
      const headers: Record<string, string> = { 'X-Neshter-Token': opts.token };
      if (body !== undefined) headers['Content-Type'] = 'application/json';
      let res: Response;
      try {
        res = await fetchFn(path, {
          method,
          headers,
          ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
          ...(signal ? { signal } : {}),
        });
      } catch (error) {
        if (error instanceof TypeError && !isAbortError(error)) throw new ApiError(0, 'network', String(error.message));
        throw error;
      }
      if (!res.ok) throw await apiErrorFromResponse(res);
      let payload: unknown;
      if (res.status !== 204) {
        try {
          payload = await res.json();
        } catch (error) {
          throw new ApiError(res.status, 'internal', `Unreadable response body: ${String(error)}`);
        }
      }
      const parsed = schema.safeParse(payload);
      if (!parsed.success) throw new ApiError(res.status, 'internal', parsed.error.issues);
      return parsed.data;
    });
  }

  const get = <S extends z.ZodType>(path: string, schema: S) => request('GET', path, schema);
  const post = <S extends z.ZodType>(path: string, schema: S, body?: unknown) => request('POST', path, schema, body);
  const jobPath = (id: string) => `/api/jobs/${encodeURIComponent(id)}`;
  const draftPath = (sha: string) => `/api/draft/${encodeURIComponent(sha)}`;

  return {
    system: () => get('/api/system', schemas.system),
    models: () => get('/api/models', schemas.models),
    concepts: () => get('/api/concepts', schemas.concepts),
    session: () => get('/api/session', schemas.session),
    loadModel: (body) => post('/api/session/load', schemas.jobStart, body),
    sample: () => post('/api/models/sample', schemas.jobStart),
    activity: (body) => post('/api/activity', schemas.activity, body),
    mlp: (body) => post('/api/mlp', schemas.mlp, body),
    lens: (body) => post('/api/lens', schemas.lens, body),
    attention: (body) => post('/api/attention', schemas.attention, body),
    graph: (body) => post('/api/graph', schemas.jobStart, body),
    graphCut: (graphId, edgeIds) =>
      post(`/api/graph/${encodeURIComponent(graphId)}/cut`, schemas.graphCut, { edge_ids: edgeIds }),
    compare: (body) => post('/api/compare', schemas.compare, body),
    recipeCheck: (body) => post('/api/recipe/check', schemas.recipeCheck, body),
    effect: (body) => post('/api/effect', schemas.jobStart, body),
    hunt: (body) => post('/api/hunt', schemas.jobStart, body),
    suite: (body) => post('/api/suite', schemas.jobStart, body),
    estimate: (body) => post('/api/export/estimate', schemas.estimate, body),
    exportModel: (body) => post('/api/export', schemas.jobStart, body),
    jobs: () => get('/api/jobs', schemas.jobs),
    // The result stays `unknown` at runtime; callers name the type they expect for the job kind.
    job: <R>(id: string, signal?: AbortSignal) =>
      request('GET', jobPath(id), schemas.job, undefined, signal) as Promise<JobState<R>>,
    cancelJob: <R>(id: string) => post(`${jobPath(id)}/cancel`, schemas.cancelJob) as Promise<JobState<R>>,
    drafts: () => get('/api/drafts', schemas.drafts),
    draft: (sha) => get(draftPath(sha), schemas.draft),
    putDraft: (sha, body) => request('PUT', draftPath(sha), schemas.putDraft, body),
    deleteDraft: (sha) => request('DELETE', draftPath(sha), schemas.deleteDraft),
    prefs: () => get('/api/prefs', schemas.prefs),
    putPrefs: (prefs) => request('PUT', '/api/prefs', schemas.putPrefs, prefs),
    shutdown: () => post('/api/shutdown', schemas.shutdown),
    streamJob: (id, { signal, onEvent }) =>
      guarded(() =>
        streamSse(`${jobPath(id)}/events`, { token: opts.token, signal, onEvent, fetchFn }),
      ),
  };
}
