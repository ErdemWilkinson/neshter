import type { JobState } from '../../schemas';
import { ApiError } from '../../errors';
import type { Bridge } from '../bridge';
import { NativeError } from '../bridge';
import { createDeviceApi, type DeviceInfo } from '../deviceApi';

const INFO: DeviceInfo = {
  id: 'bundled', kind: 'onnx',
  name: 'Qwen2.5-0.5B-Instruct', family: 'qwen2', n_layers: 4, n_heads: 3, n_kv_heads: 1, head_dim: 8, d_model: 24,
  d_mlp: 48, n_params: 1000, license: 'apache-2.0', size_bytes: 700, sha256: 'abc',
  neutral: ['a', 'b'], heads_supported: true, lens_supported: true,
};

/** A fake native side: KL is 1 for the target mask (heads cut at layer 0), `control` for any other mask. */
function fakeBridge(control = 0.1) {
  const calls: { method: string; args: unknown }[] = [];
  const store = new Map<string, unknown>();
  const bridge: Bridge = {
    async invoke<T>(method: string, args: unknown = {}): Promise<T> {
      calls.push({ method, args });
      const a = args as Record<string, unknown>;
      switch (method) {
        case 'info': return { ram_bytes: 8e9, cpu_threads: 8, active: null } as T;
        case 'models': return [{ id: 'bundled', kind: 'onnx', name: INFO.name, size_bytes: 700, info: INFO }] as T;
        case 'load': return INFO as T;
        case 'klBase': case 'cancel': return null as T;
        case 'chat': return 'hello' as T;
        case 'kl': {
          const m = a['mask'] as number[][];
          return [m[0]?.[0] === 0 ? 1 : control, m[0]?.[0] === 0 ? 1 : control] as T;
        }
        case 'probe': return { answer: { token_id: 1, text: 'x' }, p_before: 0.5, p_after: 0.2, top_before: [], top_after: [] } as T;
        case 'generate': return ((a['mask'] as number[][]).flat().includes(0) ? 'cut' : 'orig') as T;
        case 'activity': return { tokens: ['a', 'b'], power_raw: [[[1, 2], [3, 4], [5, 6]], [[0, 0], [1, 1], [2, 2]]] } as T;
        case 'storeGet': return (store.get(a['key'] as string) ?? null) as T;
        case 'storePut': store.set(a['key'] as string, a['value']); return null as T;
        case 'storeDelete': store.delete(a['key'] as string); return null as T;
        case 'storeList': return [...store.entries()].filter(([k]) => k.startsWith(a['prefix'] as string)).map(([, v]) => v) as T;
        case 'boom': throw new NativeError('no memory', 'oom');
        default: throw new NativeError(`unknown ${method}`, 'internal');
      }
    },
  };
  return { bridge, calls };
}

async function finish(api: ReturnType<typeof createDeviceApi>, id: string): Promise<JobState<unknown>> {
  let last: JobState<unknown> | null = null;
  const how = await api.streamJob(id, { signal: new AbortController().signal, onEvent: (_e, s) => (last = s as JobState<unknown>) });
  expect(how).toBe('ended');
  return last as unknown as JobState<unknown>;
}

test('nothing works before the model is opened; opening it fills the session', async () => {
  const { bridge } = fakeBridge();
  const api = createDeviceApi(bridge);
  expect(await api.session()).toEqual({ loaded: false });
  await expect(api.lens({ prompt: 'x' } as never)).rejects.toMatchObject({ code: 'no_model' });
  const { job_id } = await api.loadModel({ path: 'device:x' } as never);
  expect((await finish(api, job_id)).state).toBe('done');
  const s = (await api.session()) as { loaded: boolean; arch: { n_layers: number } };
  expect(s.loaded).toBe(true);
  expect(s.arch.n_layers).toBe(4);
  expect((await api.models())).toHaveLength(1);
});

test('desktop-only features answer with a plain sentence', async () => {
  const api = createDeviceApi(fakeBridge().bridge);
  await expect(api.graph({} as never)).rejects.toBeInstanceOf(ApiError);
  await expect(api.suite({} as never)).rejects.toMatchObject({ code: 'unsupported' });
});

test('a real effect: controls never beat the target, so it is significant with the full p', async () => {
  const { bridge, calls } = fakeBridge(0.1);
  const api = createDeviceApi(bridge);
  await finish(api, (await api.loadModel({} as never)).job_id);
  const { job_id } = await api.effect({ heads: [[0, 0]], n_max: 19, probe: 'soru' } as never);
  const end = await finish(api, job_id);
  expect(end.state).toBe('done');
  const r = end.result as Record<string, unknown>;
  expect(r).toMatchObject({ kl: 1, n_drawn: 19, k_exceed: 0, p_value: 1 / 20, significant: true, layer_matched: true, session_measure_index: 1 });
  expect(r['probe']).toMatchObject({ p_after: 0.2 });
  expect(calls.filter((c) => c.method === 'klBase')).toHaveLength(1);
  // every control is a single head of layer 0, never the selected one
  for (const c of calls.filter((x) => x.method === 'kl').slice(1)) {
    const m = (c.args as { mask: number[][] }).mask;
    expect(m.flat().filter((v) => v === 0)).toHaveLength(1);
    expect(m[0]?.[0]).toBe(1);
    expect(m[0]?.includes(0)).toBe(true);
  }
});

test('no effect: controls beat the target, the run stops early and is not significant', async () => {
  const api = createDeviceApi(fakeBridge(5).bridge);
  await finish(api, (await api.loadModel({} as never)).job_id);
  const end = await finish(api, (await api.effect({ heads: [[0, 0]] } as never)).job_id);
  expect(end.result).toMatchObject({ n_drawn: 2, stopped_early: true, significant: false, p_kind: 'besag_clifford' });
});

test('activity at a token is scaled per layer like the server', async () => {
  const api = createDeviceApi(fakeBridge().bridge);
  await finish(api, (await api.loadModel({} as never)).job_id);
  const r = (await api.activity({ prompt: 'ab', token_index: -1 } as never)) as { power: number[][] };
  expect(r.power).toEqual([[0, 0.5, 1], [0, 0.5, 1]]);
});

test('compare runs the prompt with and without the cut', async () => {
  const api = createDeviceApi(fakeBridge().bridge);
  await finish(api, (await api.loadModel({} as never)).job_id);
  const r = await api.compare({ prompt: 'p', recipe_ops: [{ type: 'head', layer: 1, head: 2 }] } as never);
  expect(r).toMatchObject({ original: 'orig', modified: 'cut' });
  await expect(api.compare({ prompt: 'p', recipe_ops: [{ type: 'head', layer: 9, head: 0 }] } as never)).rejects.toMatchObject({ code: 'invalid_input' });
});

test('prefs and drafts live in the native store', async () => {
  const api = createDeviceApi(fakeBridge().bridge);
  expect((await api.prefs()).quality).toBe('balanced');
  await api.putPrefs({ lang: 'tr', quality: 'full', reduced_motion: 'system', refusal_card_seen: true } as never);
  expect((await api.prefs()).lang).toBe('tr');
  await api.putDraft('abc', { sha256: 'abc', model_name: 'm', unsaved_count: 2, ops: [] } as never);
  expect(await api.draft('abc')).toMatchObject({ ops: [] });
  expect(await api.drafts()).toMatchObject([{ sha256: 'abc', model_name: 'm', unsaved_count: 2 }]);
  await api.deleteDraft('abc');
  await expect(api.draft('abc')).rejects.toMatchObject({ code: 'not_found' });
});

test('an aborted stream says so', async () => {
  const api = createDeviceApi(fakeBridge().bridge);
  const ctl = new AbortController();
  const { job_id } = await api.loadModel({} as never);
  const p = api.streamJob(job_id, { signal: ctl.signal, onEvent: () => undefined });
  ctl.abort();
  expect(['aborted', 'ended']).toContain(await p);
});

test('the recipe check accepts head cuts of this model only', async () => {
  const api = createDeviceApi(fakeBridge().bridge);
  await expect(api.recipeCheck({ recipe: {} })).rejects.toMatchObject({ code: 'no_model' }); // "this model" needs one open
  await finish(api, (await api.loadModel({ path: 'device:bundled' } as never)).job_id);
  const ok = await api.recipeCheck({ recipe: { source_model: { sha256: 'abc' }, interventions: [{ type: 'head', layer: 3, head: 2 }] } });
  expect(ok).toEqual({ ok: true, errors: [], sha_mismatch: false });
  const bad = await api.recipeCheck({ recipe: { source_model: { sha256: 'zzz' }, interventions: [{ type: 'head', layer: 4, head: 0 }, { type: 'direction' }] } });
  expect(bad.ok).toBe(false);
  expect(bad.errors).toHaveLength(2);
  expect(bad.sha_mismatch).toBe(true);
});

test('native error codes reach the UI as the server would send them', async () => {
  const bridge: Bridge = {
    invoke: () => Promise.reject(new NativeError('Model dosyası yok', 'not_found')),
  };
  await expect(createDeviceApi(bridge).loadModel({} as never).then(() => createDeviceApi(bridge).system())).rejects.toMatchObject({
    code: 'not_found',
  });
  const odd: Bridge = { invoke: () => Promise.reject(new NativeError('?', 'weird_code')) };
  await expect(createDeviceApi(odd).system()).rejects.toMatchObject({ code: 'internal' });
});

test('opening a model asks for that model by its id, and its info replaces the last one', async () => {
  const { bridge, calls } = fakeBridge();
  const api = createDeviceApi(bridge);
  const { job_id } = await api.loadModel({ path: 'device:SmolLM2-135M' } as never);
  expect((await finish(api, job_id)).state).toBe('done');
  expect(calls.find((c) => c.method === 'load')?.args).toEqual({ id: 'SmolLM2-135M' });
  expect(api.activeModel()?.name).toBe(INFO.name);
  const [row] = await api.models();
  expect(row).toMatchObject({ path: 'device:bundled', fmt: 'onnx', badge: 'ok', supported: true });
});

test('a GGUF whose heads cannot be read still chats, but the head views say why they are missing', async () => {
  const gguf: DeviceInfo = { ...INFO, id: 'm', kind: 'gguf', heads_supported: false, lens_supported: false };
  const { bridge, calls } = fakeBridge();
  const invoke = bridge.invoke.bind(bridge);
  bridge.invoke = (<T>(method: string, args?: unknown, onEvent?: (p: unknown) => void): Promise<T> => {
    if (method === 'load') return Promise.resolve(gguf as T);
    if (method === 'chat') {
      onEvent?.({ text: 'hel' });
      onEvent?.({ text: 'lo' });
    }
    return invoke<T>(method, args);
  }) as Bridge['invoke'];
  const api = createDeviceApi(bridge);
  await finish(api, (await api.loadModel({ path: 'device:m' } as never)).job_id);
  await expect(api.activity({ prompt: 'x' } as never)).rejects.toMatchObject({ code: 'unsupported' });
  await expect(api.lens({ prompt: 'x' } as never)).rejects.toMatchObject({ code: 'unsupported' });
  await expect(api.chat({ messages: [{ role: 'user', content: 'hi' }], cut: [[0, 0]] }, () => undefined)).rejects.toMatchObject({
    code: 'unsupported',
  });
  const seen: string[] = [];
  const answer = await api.chat({ messages: [{ role: 'user', content: 'hi' }], cut: [] }, (t) => seen.push(t));
  expect(answer).toBe('hello');
  expect(seen).toEqual(['hel', 'lo']);
  const sent = calls.find((c) => c.method === 'chat')?.args as { mask: number[][]; max_new: number };
  expect(sent.mask.flat().every((v) => v === 1)).toBe(true);
  expect(sent.max_new).toBe(512);
});
