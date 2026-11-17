// The Api of the phone app: the same interface the desktop UI talks to, answered by the model running on the
// device (through the native bridge) instead of the local server. Heavy numbers (forward passes, KL sums,
// generation) are native; the bookkeeping the server does in Python (jobs, the random-control statistics,
// session state) is here. Features that need gradients or files (connection graph, behaviour search, test set,
// saving a model) are not on the phone: they answer `invalid_input` with a plain sentence, and the UI hides them.

import type { Api, EffectRequest, JsonOp, StreamJobOptions } from '../client';
import { ApiError, toErrorCode } from '../errors';
import type * as schemas from '../schemas';
import type { DraftBody, JobState, Prefs } from '../schemas';
import type { Bridge } from './bridge';
import { NativeError } from './bridge';
import { layerMatchedHeads, percentile, seededRandom, sequentialPermutation, type Head } from './stats';

/** What the native side reports about the open model: the bundled ONNX export or a GGUF the user brought in. */
export interface DeviceInfo {
  id: string;
  kind: 'onnx' | 'gguf';
  name: string;
  family: string;
  n_layers: number;
  n_heads: number;
  n_kv_heads: number;
  head_dim: number;
  d_model: number;
  d_mlp: number;
  n_params: number;
  license: string | null;
  size_bytes: number;
  sha256: string;
  neutral: string[];
  /** false: the architecture has no per-head attention output llama.cpp exposes (only chat works). */
  heads_supported: boolean;
  lens_supported: boolean;
}

interface PhoneInfo {
  ram_bytes: number;
  cpu_threads: number;
  active: DeviceInfo | null;
}

/** One model on the phone, as `models` lists it. */
export interface PhoneModel {
  id: string;
  kind: 'onnx' | 'gguf';
  name: string;
  size_bytes: number;
  info: Partial<DeviceInfo> & { arch?: string; desc?: string };
}

export interface ChatTurn {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

/** What only the phone app has: models brought in from the phone's storage, and a conversation. */
export interface PhoneApi {
  /** Opens the system file picker and copies the chosen GGUF in; onProgress sees the bytes copied so far. */
  importModel(onProgress: (done: number, total: number) => void): Promise<PhoneModel>;
  deleteModel(id: string): Promise<void>;
  /** The open model's info, or null before one is opened. */
  activeModel(): DeviceInfo | null;
  /** An answer to the conversation, streamed through onText; `cut` heads are removed while it is written. */
  chat(
    body: { messages: ChatTurn[]; cut: readonly Head[]; maxNew?: number; temperature?: number },
    onText: (text: string) => void,
  ): Promise<string>;
  /** Stops the generation or the copy in progress. */
  stop(): Promise<void>;
}

export const asPhoneApi = (api: unknown): PhoneApi | null =>
  typeof api === 'object' && api !== null && 'importModel' in api ? (api as PhoneApi) : null;

const NOT_ON_PHONE = 'Bu özellik telefonda yok; bilgisayardaki Neşter’de kullanılabilir.';
const DEFAULT_PREFS: Prefs = { lang: null, quality: 'balanced', reduced_motion: 'system', refusal_card_seen: false };

type Job = JobState<unknown> & { created_at: string; finished_at: string | null; cancel: boolean };

const now = (): string => new Date().toISOString();

function headsFromOps(ops: readonly JsonOp[]): Head[] {
  const out: Head[] = [];
  for (const op of ops) {
    if (op.type !== 'head') throw new ApiError(422, 'invalid_input', 'Telefonda yalnızca head kesimleri uygulanabilir.');
    out.push([Number(op['layer']), Number(op['head'])]);
  }
  return out;
}

export function createDeviceApi(bridge: Bridge): Api & PhoneApi {
  let phone: PhoneInfo | null = null;
  let info: DeviceInfo | null = null;
  let loaded = false;
  let measureCount = 0;
  let baseKey: string | null = null;
  const jobs = new Map<string, Job>();
  const listeners = new Map<string, Set<(s: JobState<unknown>) => void>>();
  let nextJob = 0;

  const call = async <T>(method: string, args?: unknown, onEvent?: (payload: unknown) => void): Promise<T> => {
    try {
      return await bridge.invoke<T>(method, args, onEvent);
    } catch (error) {
      if (error instanceof NativeError) {
        // The native side speaks the server's error codes (no_model, not_found, oom, cancelled, ...).
        const code = toErrorCode(error.code) ?? 'internal';
        const status = code === 'invalid_input' ? 422 : code === 'internal' || code === 'oom' ? 500 : 409;
        throw new ApiError(status, code, error.message);
      }
      throw error;
    }
  };
  const getPhone = async (): Promise<PhoneInfo> => (phone ??= await call<PhoneInfo>('info'));
  const getInfo = (): Promise<DeviceInfo> =>
    info === null ? Promise.reject(new ApiError(409, 'no_model', 'Önce bir model aç.')) : Promise.resolve(info);
  const requireLoaded = () => {
    if (!loaded) throw new ApiError(409, 'no_model', 'Önce bir model aç.');
  };
  const requireHeads = (i: DeviceInfo) => {
    if (!i.heads_supported) {
      throw new ApiError(422, 'unsupported', 'Bu modelin mimarisinde head’ler ayrı okunamıyor; telefonda yalnız sohbet edilebilir.');
    }
  };
  const unsupported = (): Promise<never> => Promise.reject(new ApiError(422, 'unsupported', NOT_ON_PHONE));

  const mask = (i: DeviceInfo, heads: readonly Head[]): number[][] => {
    const m = Array.from({ length: i.n_layers }, () => Array<number>(i.n_heads).fill(1));
    for (const [l, h] of heads) {
      const row = m[l];
      if (row === undefined || h < 0 || h >= i.n_heads) throw new ApiError(422, 'invalid_input', `K${l}·H${h} bu modelde yok.`);
      row[h] = 0;
    }
    return m;
  };

  const publicState = (j: Job): JobState<unknown> => {
    return {
      id: j.id, kind: j.kind, state: j.state, progress: j.progress, message: j.message, result: j.result,
      error: j.error, error_code: j.error_code, started_at: j.started_at,
    };
  };
  const update = (j: Job, patch: Partial<Job>) => {
    Object.assign(j, patch);
    for (const fn of listeners.get(j.id) ?? []) fn(publicState(j));
  };
  const isTerminal = (s: JobState<unknown>['state']) => s === 'done' || s === 'error' || s === 'cancelled';

  function startJob(kind: string, run: (j: Job) => Promise<unknown>): { job_id: string } {
    const id = `d${++nextJob}`;
    const j: Job = {
      id, kind, state: 'running', progress: 0, message: '', result: null, error: null, error_code: null,
      started_at: now(), created_at: now(), finished_at: null, cancel: false,
    };
    jobs.set(id, j);
    void (async () => {
      try {
        const result = await run(j);
        if (j.cancel) update(j, { state: 'cancelled', finished_at: now() });
        else update(j, { state: 'done', progress: 1, result, finished_at: now() });
      } catch (error) {
        if (j.cancel) update(j, { state: 'cancelled', finished_at: now() });
        else {
          const code = error instanceof ApiError ? error.code : 'internal';
          const detail = error instanceof ApiError ? String(error.detail ?? error.message) : String(error);
          update(j, { state: 'error', error: detail, error_code: code, finished_at: now() });
        }
      }
    })();
    return { job_id: id };
  }
  const checkCancel = (j: Job) => {
    if (j.cancel) throw new Error('cancelled');
  };

  async function effect(j: Job, body: EffectRequest) {
    const i = await getInfo();
    requireLoaded();
    const seen = new Set<string>();
    const heads = body.heads.filter(([l, h]) => !seen.has(`${l}:${h}`) && seen.add(`${l}:${h}`)) as Head[];
    const nMax = body.n_max ?? 39;
    const seed = body.seed ?? 0;
    const prompts = body.prompts ?? i.neutral;
    if (prompts.length === 0) throw new ApiError(422, 'invalid_input', 'En az bir prompt gerekli.');
    const rnd = seededRandom(seed);
    const controls: Head[][] = [];
    let layerMatched = true;
    for (let k = 0; k < nMax; k++) {
      const { picked, matched } = layerMatchedHeads(heads, i.n_layers, i.n_heads, rnd);
      controls.push(picked);
      layerMatched &&= matched;
    }
    const key = JSON.stringify(prompts);
    if (baseKey !== key) {
      update(j, { message: 'Orijinal cevaplar hesaplanıyor' });
      await call('klBase', { prompts });
      baseKey = key;
    }
    checkCancel(j);
    const perPrompt = await call<number[]>('kl', { mask: mask(i, heads) });
    const kl = perPrompt.reduce((s, v) => s + v, 0) / perPrompt.length;
    const klStd = Math.sqrt(perPrompt.reduce((s, v) => s + (v - kl) ** 2, 0) / perPrompt.length);
    update(j, { progress: 0.05, message: 'Seçili müdahalenin etkisi ölçüldü' });
    const out = await sequentialPermutation(
      kl,
      async (k) => {
        checkCancel(j);
        const values = await call<number[]>('kl', { mask: mask(i, controls[k] ?? []) });
        update(j, { progress: 0.05 + (0.95 * (k + 1)) / nMax, message: `Rastgele kontrol ${k + 1}/${nMax} (erken durabilir)` });
        return values.reduce((s, v) => s + v, 0) / values.length;
      },
      nMax,
    );
    const r = out.seen;
    const band = r.length >= 2;
    measureCount += 1;
    const probe = body.probe === undefined ? {} : { probe: await call('probe', { text: body.probe, mask: mask(i, heads) }) };
    return {
      kl, kl_std: klStd, kl_se: klStd / Math.sqrt(prompts.length), n_prompts: prompts.length,
      random_kls: r, random_kl: band ? r.reduce((s, v) => s + v, 0) / r.length : null,
      random_p05: band ? percentile(r, 5) : null, random_p95: band ? percentile(r, 95) : null,
      percentile: band ? (100 * r.filter((v) => v < kl).length) / r.length : null,
      n_drawn: out.nDrawn, n_max: nMax, k_exceed: out.kExceed, tau: 0.05, p_value: out.pValue, p_kind: out.pKind,
      p_min: 1 / (nMax + 1), significant: out.significant, stopped_early: out.stoppedEarly, layer_matched: layerMatched,
      seed, session_measure_index: measureCount, ...probe,
    };
  }

  const sessionInfo = async (): Promise<schemas.SessionInfo> => {
    if (!loaded) return { loaded: false };
    const i = await getInfo();
    return {
      loaded: true, name: i.name, device: 'phone', dtype: i.kind === 'gguf' ? 'gguf' : 'int8', license: i.license, memory_bytes: i.size_bytes,
      device_note: null, source_path: `device:${i.name}`, source_sha256: i.sha256, default_method: 'activation_diff',
      safety_dirs: 0, safety_heads: 0, measure_count: measureCount,
      arch: { family: i.family, n_layers: i.n_layers, n_heads: i.n_heads, n_kv_heads: i.n_kv_heads, head_dim: i.head_dim, d_model: i.d_model, d_mlp: i.d_mlp, n_params: i.n_params },
    } as schemas.SessionInfo;
  };

  const normalize = (rows: number[][], scale: 'layer' | 'global'): number[][] => {
    const flat = rows.flat();
    const span = (xs: number[]) => [Math.min(...xs), Math.max(...xs)] as const;
    if (scale === 'global') {
      const [lo, hi] = span(flat);
      return rows.map((r) => r.map((v) => (hi > lo ? (v - lo) / (hi - lo) : 0)));
    }
    return rows.map((r) => {
      const [lo, hi] = span(r);
      return r.map((v) => (hi > lo ? (v - lo) / (hi - lo) : 0));
    });
  };

  return {
    async system() {
      const p = await getPhone();
      return {
        gpu_name: null, vram_bytes: 0, ram_bytes: p.ram_bytes, default_out_dir: '',
        llama_cpp: { found: true, path: null, converter: false }, ollama: false, torch_version: 'llama.cpp',
        torch_cuda: null, cuda_available: false, native_bf16: false, cpu_threads: p.cpu_threads,
      };
    },
    async models() {
      const p = await getPhone();
      const list = await call<PhoneModel[]>('models');
      return list.map((m) => {
        const a = m.info;
        // A model larger than about 70% of the phone's memory will be slow or fail to open.
        const badge = m.size_bytes > p.ram_bytes * 0.7 ? 'no' : m.size_bytes > p.ram_bytes * 0.4 ? 'slow' : 'ok';
        return {
          name: m.name, source: m.kind === 'gguf' ? 'GGUF' : 'phone', path: `device:${m.id}`, fmt: m.kind,
          size_bytes: m.size_bytes, memory_bytes: m.size_bytes, badge, license: a.license ?? null,
          arch: {
            family: a.family ?? a.arch ?? '?', n_layers: a.n_layers ?? null, n_heads: a.n_heads ?? null,
            n_kv_heads: a.n_kv_heads ?? null, d_mlp: a.d_mlp ?? null,
          },
          supported: true, unsupported_reason: null,
        };
      });
    },
    concepts: async () => [],
    session: sessionInfo,
    loadModel(body) {
      // No path: the native side opens the model opened last (or the bundled one).
      const path = (body as Partial<typeof body> | undefined)?.path ?? '';
      const id = path.startsWith('device:') ? path.slice('device:'.length) : undefined;
      return Promise.resolve(startJob('load', async (j) => {
        update(j, { message: 'Model açılıyor' });
        loaded = false;
        info = await call<DeviceInfo>('load', id === undefined ? {} : { id });
        loaded = true;
        baseKey = null; // the KL baseline belonged to the model that was open
        const i = info;
        return { arch: (await sessionInfo() as { arch: unknown }).arch, name: i.name, license: i.license, device: 'phone', device_note: null, dtype: 'int8', source_sha256: i.sha256 };
      }));
    },
    sample: unsupported,
    async activity(body) {
      requireLoaded();
      requireHeads(await getInfo());
      const res = await call<{ tokens: string[]; power_raw: number[][][] }>('activity', { text: body.prompt });
      if (body.token_index == null) return res;
      const t = body.token_index < 0 ? res.tokens.length + body.token_index : body.token_index;
      const at = res.power_raw.map((layer) => layer.map((head) => head[t] ?? 0));
      return { ...res, power: normalize(at, body.scale ?? 'layer') };
    },
    mlp: unsupported,
    async lens(body) {
      requireLoaded();
      if (!(await getInfo()).lens_supported) {
        throw new ApiError(422, 'unsupported', 'Bu modelin mimarisinde katman çıktıları okunamıyor; katman katman tahmin yok, ısı haritası ve dikkat görünümü var.');
      }
      return call('lens', { text: body.prompt, position: body.position ?? -1 });
    },
    async attention(body) {
      requireLoaded();
      requireHeads(await getInfo());
      return call('attention', { text: body.prompt, layer: body.layer });
    },
    graph: unsupported,
    graphCut: unsupported,
    async compare(body) {
      requireLoaded();
      const i = await getInfo();
      const heads = headsFromOps(body.recipe_ops);
      // Same limit as the desktop (the UI's "shortened" note names it); the engine's key/value cache keeps it quick.
      const maxNew = body.max_new_tokens ?? 200;
      const original = await call<string>('generate', { prompt: body.prompt, mask: mask(i, []), max_new: maxNew });
      const modified = await call<string>('generate', { prompt: body.prompt, mask: mask(i, heads), max_new: maxNew });
      return { original, modified, mode: 'sequential', target_hit: null };
    },
    async recipeCheck({ recipe }) {
      // The phone's version of /api/recipe/check: head cuts only, inside this model, for this model's weights.
      const i = await getInfo();
      const r = (recipe ?? {}) as { source_model?: { sha256?: string }; interventions?: JsonOp[] };
      const errors: string[] = [];
      for (const op of r.interventions ?? []) {
        const l = Number(op['layer']);
        const h = Number(op['head']);
        if (op.type !== 'head') errors.push(`${op.type}: telefonda yalnızca head kesimleri var.`);
        else if (!(l >= 0 && l < i.n_layers && h >= 0 && h < i.n_heads)) errors.push(`K${l}·H${h} bu modelde yok.`);
      }
      const shaMismatch = r.source_model?.sha256 !== undefined && r.source_model.sha256 !== i.sha256;
      return { ok: errors.length === 0, errors, sha_mismatch: shaMismatch };
    },
    async effect(body) {
      requireLoaded();
      requireHeads(await getInfo());
      return startJob('effect', (j) => effect(j, body));
    },
    hunt: unsupported,
    suite: unsupported,
    estimate: unsupported,
    exportModel: unsupported,
    async jobs() {
      return [...jobs.values()].map((j) => ({ ...publicState(j), created_at: j.created_at, finished_at: j.finished_at }));
    },
    async job<R>(id: string) {
      const j = jobs.get(id);
      if (j === undefined) throw new ApiError(404, 'not_found', id);
      return publicState(j) as JobState<R>;
    },
    async cancelJob<R>(id: string) {
      const j = jobs.get(id);
      if (j === undefined) throw new ApiError(404, 'not_found', id);
      j.cancel = true;
      await call('cancel').catch(() => undefined);
      return publicState(j) as JobState<R>;
    },
    async drafts() {
      const all = await call<DraftBody[]>('storeList', { prefix: 'draft-' });
      return all.map((d) => ({ sha256: d.sha256, model_name: d.model_name, updated_at: d.updated_at, unsaved_count: d.unsaved_count }));
    },
    async draft(sha) {
      const d = await call<DraftBody | null>('storeGet', { key: `draft-${sha}` });
      if (d === null) throw new ApiError(404, 'not_found', sha);
      return d;
    },
    async putDraft(sha, body) {
      const d = { ...body, updated_at: now() } as DraftBody;
      await call('storePut', { key: `draft-${sha}`, value: d });
      return d;
    },
    async deleteDraft(sha) {
      await call('storeDelete', { key: `draft-${sha}` });
    },
    async prefs() {
      return (await call<Prefs | null>('storeGet', { key: 'prefs' })) ?? DEFAULT_PREFS;
    },
    async putPrefs(prefs) {
      await call('storePut', { key: 'prefs', value: prefs });
      return prefs;
    },
    shutdown: async () => ({ ok: true }),
    async importModel(onProgress) {
      const i = await call<PhoneModel['info'] & { file_bytes?: number }>('importModel', {}, (e) => {
        const { done, total } = e as { done: number; total: number };
        onProgress(done, total);
      });
      return { id: String(i.id), kind: 'gguf', name: String(i.name), size_bytes: Number(i.file_bytes ?? 0), info: i };
    },
    async deleteModel(id) {
      await call('deleteModel', { id });
      if (info?.id === id) {
        info = null;
        loaded = false;
      }
    },
    activeModel: () => info,
    async chat(body, onText) {
      requireLoaded();
      const i = await getInfo();
      if (body.cut.length > 0) requireHeads(i);
      return call<string>(
        'chat',
        { messages: body.messages, mask: mask(i, body.cut), max_new: body.maxNew ?? 512, temperature: body.temperature ?? 0 },
        (e) => onText((e as { text: string }).text),
      );
    },
    async stop() {
      await call('cancel');
    },
    streamJob(id: string, opts: StreamJobOptions) {
      return new Promise((resolve) => {
        const j = jobs.get(id);
        if (j === undefined) return resolve('ended');
        const set = listeners.get(id) ?? new Set();
        listeners.set(id, set);
        const done = (how: 'ended' | 'aborted') => {
          set.delete(fn);
          opts.signal.removeEventListener('abort', onAbort);
          resolve(how);
        };
        const fn = (s: JobState<unknown>) => {
          opts.onEvent('status', s);
          if (isTerminal(s.state)) done('ended');
        };
        const onAbort = () => done('aborted');
        opts.signal.addEventListener('abort', onAbort);
        set.add(fn);
        fn(publicState(j));
      });
    },
  };
}
