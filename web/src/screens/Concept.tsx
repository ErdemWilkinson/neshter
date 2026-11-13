import { useEffect, useRef, useState } from 'react';
import { useStore } from 'zustand';
import type { HuntRequest } from '@/api/client';
import { getApi } from '@/api/currentApi';
import { ApiError, BusyError, isAbortError, type ErrorCode } from '@/api/errors';
import { isTerminal } from '@/api/jobs';
import { huntResult, type Concept as ConceptInfo, type HuntResult } from '@/api/schemas';
import { useJob } from '@/api/useJob';
import { loadPrefs, savePrefs } from '@/app/prefs';
import { isLocked, jobsStore } from '@/core/stores/jobs';
import { sessionStore } from '@/core/stores/session';
import { BasketPanel } from '@/features/basket/BasketPanel';
import { ExampleEditor, MIN_EXAMPLES, parseLines } from '@/features/concept/ExampleEditor';
import { FoundDirection } from '@/features/concept/FoundDirection';
import { FoundHeads } from '@/features/concept/FoundHeads';
import { LayerChart, ValidationLine } from '@/features/concept/LayerChart';
import { MethodPicker, type HuntMethod } from '@/features/concept/MethodPicker';
import { SweepField } from '@/features/concept/SweepField';
import { CUSTOM, PresetChips } from '@/features/concept/PresetChips';
import { RefusalCard } from '@/features/concept/RefusalCard';
import { slugify } from '@/features/concept/slug';
import { useT } from '@/i18n';
import { Button } from '@/ui/Button';
import { ErrorCard } from '@/ui/ErrorCard';
import { ProgressCard } from '@/ui/ProgressCard';

interface Failure {
  code: ErrorCode;
  detail: unknown;
}

const asFailure = (error: unknown): Failure =>
  error instanceof ApiError ? { code: error.code, detail: error.detail } : { code: 'internal', detail: String(error) };

interface Found {
  result: HuntResult;
  /** The concept the basket records for a direction found by this search (`refusal` for a safety concept). */
  conceptId: string;
}

/** A custom concept's id: a slug that does not collide with a preset (a preset id is never reused). */
function customId(name: string, concepts: readonly ConceptInfo[]): string {
  const slug = slugify(name);
  const taken = concepts.some((concept) => concept.id.toLowerCase() === slug.toLowerCase());
  return taken ? `${slug.slice(0, 58)}-ozel` : slug;
}

/** Spec §2 step 3b: describe a behaviour with examples, let Neşter find the heads or the direction that carry it. */
export function Concept() {
  const t = useT();
  const model = useStore(sessionStore, (s) => s.model);
  const locked = useStore(jobsStore, isLocked);
  const { job, start, cancel, reset } = useJob<HuntResult>();

  const [concepts, setConcepts] = useState<ConceptInfo[] | null>(null);
  const [loadFailure, setLoadFailure] = useState<Failure | null>(null);
  const [picked, setPicked] = useState<string | null>(null);
  const [name, setName] = useState('');
  const [customPositive, setCustomPositive] = useState('');
  const [customNegative, setCustomNegative] = useState('');
  const [method, setMethod] = useState<HuntMethod>(model?.default_method ?? 'attribution');
  const [sweeps, setSweeps] = useState(1);
  const [cardSeen, setCardSeen] = useState<boolean | null>(null);
  const [found, setFound] = useState<Found | null>(null);
  const [failure, setFailure] = useState<Failure | null>(null);
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    const api = getApi();
    api
      .concepts()
      .then((list) => alive.current && setConcepts(list))
      .catch((error: unknown) => alive.current && setLoadFailure(asFailure(error)));
    void loadPrefs(api).then((prefs) => alive.current && setCardSeen(prefs.refusal_card_seen));
    return () => {
      alive.current = false;
    };
  }, []);

  const selected = picked ?? concepts?.find((concept) => concept.id === 'code')?.id ?? concepts?.[0]?.id ?? CUSTOM;
  const preset = concepts?.find((concept) => concept.id === selected) ?? null;
  const isCustom = preset === null;
  const positive = isCustom ? customPositive : (preset?.positive ?? []).join('\n');
  const negative = isCustom ? customNegative : (preset?.negative ?? []).join('\n');
  const enough = parseLines(positive).length >= MIN_EXAMPLES && parseLines(negative).length >= MIN_EXAMPLES;
  const running = job !== null && !isTerminal(job);
  const showRefusalCard = preset?.safety_ablation === true && cardSeen === false;
  const canSearch = enough && concepts !== null && !locked && !running;

  const acknowledge = () => {
    setCardSeen(true);
    // A failed save only means the card shows once more next time.
    void savePrefs(getApi(), { refusal_card_seen: true }).catch(() => undefined);
  };

  /** The request body and the concept the basket will record for what this search finds. */
  const buildRequest = (): { body: HuntRequest; conceptId: string } => {
    if (preset !== null) {
      const extra = method === 'direction' && sweeps > 1 ? { sweeps } : {};
      return { body: { concept_id: preset.id, method, ...extra }, conceptId: preset.safety_ablation ? 'refusal' : preset.id };
    }
    const id = customId(name, concepts ?? []);
    const label = name.trim() === '' ? id : name.trim();
    const concept = { id, name_tr: label, name_en: label, positive: parseLines(positive), negative: parseLines(negative) };
    const extra = method === 'direction' && sweeps > 1 ? { sweeps } : {};
    return { body: { concept, method, ...extra }, conceptId: id };
  };

  const search = async () => {
    const { body, conceptId } = buildRequest();
    setFound(null);
    setFailure(null);
    try {
      const final = await start(() => getApi().hunt(body));
      if (!alive.current) return;
      if (final.state === 'cancelled') return reset();
      if (final.state !== 'done') return setFailure({ code: final.error_code ?? 'internal', detail: final.error });
      const parsed = huntResult.safeParse(final.result);
      if (!parsed.success) return setFailure({ code: 'internal', detail: parsed.error.issues });
      setFound({ result: parsed.data, conceptId });
    } catch (error) {
      if (isAbortError(error) || !alive.current) return;
      // Busy is the banner's business (the client flags it too; this also covers a client without the hook).
      if (error instanceof BusyError) sessionStore.getState().setBusy(true);
      else setFailure(asFailure(error));
    }
  };

  return (
    <section className="concept-screen">
      <h1>{t('app.step.concept')}</h1>
      <p className="concept-lead">{t('concept.lead')}</p>

      {loadFailure !== null && <ErrorCard code={loadFailure.code} detail={loadFailure.detail} />}
      {concepts !== null && <PresetChips concepts={concepts} value={selected} onChange={setPicked} />}
      {showRefusalCard && <RefusalCard onAcknowledge={acknowledge} />}

      {concepts !== null && (
        <>
          {isCustom && (
            <label className="concept-name">
              <span>{t('concept.name')}</span>
              <input type="text" value={name} maxLength={80} onChange={(event) => setName(event.target.value)} />
            </label>
          )}
          <ExampleEditor
            positive={positive}
            negative={negative}
            readOnly={!isCustom}
            onPositive={setCustomPositive}
            onNegative={setCustomNegative}
          />
          <MethodPicker value={method} onChange={setMethod} />
          {method === 'direction' && <SweepField value={sweeps} onChange={setSweeps} />}
          <Button variant="primary" disabled={!canSearch} onClick={() => void search()}>
            {t('concept.search')}
          </Button>
        </>
      )}

      {running && <ProgressCard job={job} onCancel={() => void cancel()} />}
      {failure !== null && (
        <>
          <p>{t('concept.failed')}</p>
          <ErrorCard code={failure.code} detail={failure.detail} />
        </>
      )}

      {found !== null && (
        <div className="concept-result">
          <ValidationLine result={found.result} />
          <FoundHeads heads={found.result.heads} />
          <FoundDirection result={found.result} conceptId={found.conceptId} />
          <p className="concept-note">{t('concept.result.note')}</p>
          <LayerChart scores={found.result.layer_scores} refLayer={found.result.ref_layer} />
        </div>
      )}

      <BasketPanel />
    </section>
  );
}
