import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useStore } from 'zustand';
import type { JsonOp } from '@/api/client';
import { getApi } from '@/api/currentApi';
import { ApiError, BusyError, isAbortError, type ErrorCode } from '@/api/errors';
import { exportResult, type Concept, type ExportEstimate } from '@/api/schemas';
import { useJob } from '@/api/useJob';
import { countSavable, savableOps } from '@/core/basket';
import { basketConcept } from '@/core/basketConcept';
import { isSavable } from '@/core/ops';
import { basketStore } from '@/core/stores/basket';
import { isLocked, jobsStore } from '@/core/stores/jobs';
import { sessionStore } from '@/core/stores/session';
import { formatBytes } from '@/features/models/format';
import { saveFlowStore } from '@/features/saveflow/saveFlowStore';
import { useSaveFlow } from '@/features/saveflow/useSaveFlow';
import { useT, type MessageKey } from '@/i18n';
import { Button } from '@/ui/Button';
import { ErrorCard } from '@/ui/ErrorCard';
import { ProgressCard } from '@/ui/ProgressCard';
import { DEFAULT_FORMAT, formatOptions, isGguf, ollamaAvailable, type FormatId, type FormatNote } from './formats';
import { checkName, type NameCheck } from './nameRule';
import { checkOutDir, joinOutDir, type OutDirCheck } from './outDir';

const NAME_MESSAGES: Record<Exclude<NameCheck, 'ok'>, MessageKey> = {
  empty: 'save.name.empty',
  chars: 'save.name.chars',
  reserved: 'save.name.reserved',
  trailing: 'save.name.trailing',
};
const DIR_MESSAGES: Record<Exclude<OutDirCheck, 'ok'>, MessageKey> = {
  relative: 'save.dir.relative',
  unc: 'save.dir.unc',
  control: 'save.dir.control',
};
const FORMAT_LABELS: Record<FormatId, MessageKey> = {
  safetensors: 'save.format.safetensors',
  gguf_q4_k_m: 'save.format.gguf_q4_k_m',
  gguf_q8_0: 'save.format.gguf_q8_0',
};
const NOTE_LABELS: Record<Exclude<FormatNote, null>, MessageKey> = {
  experimental: 'save.format.experimental',
  needs_llama: 'save.format.needs_llama',
};

const ESTIMATE_DEBOUNCE_MS = 300;

interface Failure {
  code: ErrorCode;
  detail: unknown;
}

/**
 * The export form. The Save button only calls `saveClicked()`; the export itself starts when the save flow reaches
 * `exportRequested`, so the explore-only discard and the refreshed Compare screen always come first.
 */
export function SaveForm() {
  const t = useT();
  const flow = useSaveFlow();
  const items = useStore(basketStore, (s) => s.items);
  const system = useStore(sessionStore, (s) => s.system);
  const modelBusy = useStore(sessionStore, (s) => s.busy);
  const active = useStore(jobsStore, (s) => s.active);
  const locked = useStore(jobsStore, isLocked);
  const [concepts, setConcepts] = useState<Concept[]>([]);
  const [format, setFormat] = useState<FormatId>(DEFAULT_FORMAT);
  const [ollama, setOllama] = useState(false);
  const [nameInput, setNameInput] = useState<string | null>(null);
  const [suggested, setSuggested] = useState('');
  const [dirInput, setDirInput] = useState<string | null>(null);
  const [estimate, setEstimate] = useState<ExportEstimate | null>(null);
  const [failure, setFailure] = useState<Failure | null>(null);
  const { start } = useJob<unknown>();
  const alive = useRef(true);
  const startedFor = useRef<unknown>(null);

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  useEffect(() => {
    let live = true;
    const api = getApi();
    api
      .concepts()
      .then((list) => live && setConcepts(list))
      .catch(() => undefined); // without concepts there is simply no concept label
    if (sessionStore.getState().system === null) {
      api
        .system()
        .then((info) => live && sessionStore.getState().setSystem(info))
        .catch(() => undefined);
    }
    return () => {
      live = false;
    };
  }, []);

  const ops = useMemo(() => savableOps(items), [items]);
  const { conceptId } = useMemo(
    () =>
      basketConcept(
        items.filter((item) => isSavable(item.op)),
        concepts,
      ),
    [items, concepts],
  );
  const defaultDir = system?.default_out_dir ?? '';
  const dir = dirInput ?? defaultDir;
  const name = nameInput ?? suggested;
  const nameCode = checkName(name);
  const dirCode = checkOutDir(dir, defaultDir);
  const options = formatOptions(system);
  const showOllama = ollamaAvailable(system, format);

  useEffect(() => {
    if (dirCode !== 'ok') return;
    let live = true;
    const timer = setTimeout(() => {
      getApi()
        .estimate({ format, recipe_ops: ops as JsonOp[], out_dir: dir, concept: conceptId })
        .then((result) => {
          if (!live) return;
          setEstimate(result);
          setSuggested((current) => (current === '' ? result.suggested_name : current)); // pre-fills once
        })
        .catch(() => live && setEstimate(null)); // no estimate is not a reason to block the save
    }, ESTIMATE_DEBOUNCE_MS);
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [format, ops, dir, dirCode, conceptId]);

  const shownEstimate = dirCode === 'ok' ? estimate : null;

  const runExport = useCallback(async () => {
    const flowStore = saveFlowStore.getState();
    setFailure(null);
    const fail = (cause: Failure | null) => {
      flowStore.dispatch({ type: 'EXPORT_FAILED' });
      if (cause !== null && alive.current) setFailure(cause);
    };
    try {
      const final = await start(async () => {
        const started = await getApi().exportModel({
          recipe_ops: ops as JsonOp[],
          name,
          format,
          ollama: showOllama && ollama,
          out_dir: joinOutDir(dir, name),
          concept: conceptId,
        });
        flowStore.dispatch({ type: 'EXPORT_STARTED', jobId: started.job_id });
        return started;
      });
      const parsed = exportResult.safeParse(final.result);
      if (final.state === 'done' && parsed.success) {
        basketStore.getState().markSaved();
        flowStore.dispatch({ type: 'EXPORT_DONE', result: parsed.data });
      } else if (final.state === 'done') {
        fail({ code: 'internal', detail: parsed.error?.issues });
      } else if (final.state === 'cancelled') {
        fail(null);
      } else {
        fail({ code: final.error_code ?? 'internal', detail: final.error });
      }
    } catch (error) {
      if (isAbortError(error)) return;
      if (error instanceof BusyError) {
        sessionStore.getState().setBusy(true);
        fail(null);
      } else {
        fail(error instanceof ApiError ? { code: error.code, detail: error.detail } : { code: 'internal', detail: String(error) });
      }
    }
  }, [start, ops, name, format, showOllama, ollama, dir, conceptId]);

  const requested = flow.state.kind === 'exportRequested';
  useEffect(() => {
    if (!requested || startedFor.current === flow.state) return;
    startedFor.current = flow.state;
    void runExport();
  }, [requested, flow.state, runExport]);

  const idle = flow.state.kind === 'idle' || flow.state.kind === 'reviewReady';
  const working = flow.state.kind === 'exporting' || flow.state.kind === 'exportRequested' || flow.state.kind === 'recomparing';
  const canSave = idle && nameCode === 'ok' && dirCode === 'ok' && shownEstimate?.enough !== false && !locked && !modelBusy;
  const label =
    shownEstimate === null
      ? t('save.button')
      : t('save.button.estimate', {
          needed: formatBytes(shownEstimate.bytes_needed),
          free: formatBytes(shownEstimate.free_bytes),
        });

  return (
    <form
      className="save-form"
      onSubmit={(event) => {
        event.preventDefault();
        if (canSave) flow.saveClicked();
      }}
    >
      <p>{t('save.count', { n: countSavable(items) })}</p>

      <div className="field">
        <label htmlFor="save-name">{t('save.name.label')}</label>
        <input
          id="save-name"
          type="text"
          value={name}
          maxLength={64}
          autoComplete="off"
          spellCheck={false}
          aria-invalid={nameCode !== 'ok'}
          aria-describedby="save-name-msg"
          onChange={(event) => setNameInput(event.target.value)}
        />
        <p id="save-name-msg" className="field-error">
          {nameCode === 'ok' ? '' : t(NAME_MESSAGES[nameCode])}
        </p>
      </div>

      <div className="field">
        <label htmlFor="save-dir">{t('save.dir.label')}</label>
        <input
          id="save-dir"
          type="text"
          value={dir}
          autoComplete="off"
          spellCheck={false}
          aria-invalid={dirCode !== 'ok'}
          aria-describedby="save-dir-msg"
          onChange={(event) => setDirInput(event.target.value)}
        />
        <p id="save-dir-msg" className="field-error">
          {dirCode === 'ok' ? '' : t(DIR_MESSAGES[dirCode])}
        </p>
      </div>

      <fieldset className="save-formats">
        <legend>{t('save.format.legend')}</legend>
        {options.map((option) => (
          <label key={option.id} className="save-format">
            <input
              type="radio"
              name="save-format"
              value={option.id}
              checked={format === option.id}
              disabled={!option.enabled}
              onChange={() => {
                setFormat(option.id);
                if (!isGguf(option.id)) setOllama(false);
              }}
            />
            {t(FORMAT_LABELS[option.id])}
            {option.note !== null && <span className="badge">{t(NOTE_LABELS[option.note])}</span>}
          </label>
        ))}
        <label className="save-format">
          <input
            type="checkbox"
            checked={showOllama && ollama}
            disabled={!showOllama}
            onChange={(event) => setOllama(event.target.checked)}
          />
          {t('save.ollama')}
        </label>
      </fieldset>

      <p>{t('save.info.recipe')}</p>
      <p>{t('save.info.source')}</p>
      {shownEstimate?.safety_ablated === true && <p className="badge badge-safety">{t('save.safety')}</p>}
      {shownEstimate?.enough === false && <p className="field-error">{t('errors.disk_full')}</p>}

      <Button type="submit" variant="primary" disabled={!canSave}>
        {label}
      </Button>

      {working && active !== null && (
        <ProgressCard
          job={active}
          onCancel={() => {
            if (flow.state.kind === 'recomparing') flow.cancel();
            else void getApi().cancelJob(active.id).catch(() => undefined);
          }}
        />
      )}
      {failure !== null && (
        <>
          <p>{t('save.failed')}</p>
          <ErrorCard code={failure.code} detail={failure.detail} />
        </>
      )}
    </form>
  );
}
