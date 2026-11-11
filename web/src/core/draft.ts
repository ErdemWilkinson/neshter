// Draft codec: the basket as the server stores it (`DraftIn`). The server treats basket/shelf/saved_ops as opaque
// lists, so what goes in is decided here: ops, prompts and the model name/SHA only. Never `source_path`, never
// model output, never measurements.
import { MAX_ITEMS, unsavedCount, type BasketItem, type Origin } from './basket';
import { isOpaque, opSemantics, opToJson, parseOp } from './ops';
import { z } from './zod';

export const DRAFT_SCHEMA_VERSION = 1;
const MAX_PROMPTS = 20;
const MAX_PROMPT_CHARS = 8000;
const MAX_MODEL_NAME = 256;

/** Body of `PUT /api/draft/{sha}` (without `updated_at`). A type alias, so it is assignable to the client's index-signature body. */
export type DraftPayload = {
  schema_version: 1;
  sha256: string;
  model_name: string;
  unsaved_count: number;
  basket: Record<string, unknown>[];
  shelf: Record<string, unknown>[];
  saved_ops: string[];
  prompts: string[];
};

export interface DraftInput {
  items: readonly BasketItem[];
  shelf: readonly BasketItem[];
  savedOps: readonly string[];
  model: { name: string; sha256: string };
  prompts: readonly string[];
}

/** Cut at `max` UTF-16 units without leaving half of a surrogate pair (the server rejects lone surrogates). */
export function clip(text: string, max: number): string {
  if (text.length <= max) return text;
  const end = /[\ud800-\udbff]/.test(text.charAt(max - 1)) ? max - 1 : max;
  return text.slice(0, end);
}

/** Explicit field list: anything else on an item (a measurement, stray text) cannot leak into a draft. */
function itemToJson(item: BasketItem): Record<string, unknown> {
  return {
    id: item.id,
    op: opToJson(item.op),
    semantics: item.semantics,
    origin: item.origin,
    concept: item.concept,
    addedAt: item.addedAt,
  };
}

/** Caps mirror `DraftIn`, so autosave never earns a 422; `pendingDiscard` is a transient flag and is dropped. */
export function encodeDraft(input: DraftInput): DraftPayload {
  return {
    schema_version: DRAFT_SCHEMA_VERSION,
    sha256: input.model.sha256,
    model_name: clip(input.model.name, MAX_MODEL_NAME),
    unsaved_count: unsavedCount(input.items, input.savedOps),
    basket: input.items.slice(-MAX_ITEMS).map(itemToJson),
    shelf: input.shelf.slice(-MAX_ITEMS).map(itemToJson),
    saved_ops: input.savedOps.slice(-MAX_ITEMS),
    prompts: input.prompts.slice(-MAX_PROMPTS).map((p) => clip(p, MAX_PROMPT_CHARS)),
  };
}

const ORIGINS: readonly string[] = ['heatmap', 'concept', 'graph', 'import'] satisfies Origin[];

const WireItem = z.looseObject({
  id: z.string().min(1),
  op: z.record(z.string(), z.unknown()),
  origin: z.string(),
  concept: z.string().nullish(),
  addedAt: z.string(),
});
const WireDraft = z.looseObject({
  basket: z.array(z.unknown()),
  shelf: z.array(z.unknown()),
  saved_ops: z.array(z.string()),
  prompts: z.array(z.string()),
});

export type DecodedDraft =
  | {
      ok: true;
      items: BasketItem[];
      shelf: BasketItem[];
      savedOps: string[];
      prompts: string[];
      /** Basket items whose op this version cannot validate (kept as opaque rows, never sent anywhere). */
      unsupported: number;
    }
  | { ok: false; reason: 'schema_version' | 'invalid' };

function toItem(entry: unknown): BasketItem | null {
  const parsed = WireItem.safeParse(entry);
  if (!parsed.success) return null;
  const w = parsed.data;
  const op = parseOp(w.op);
  return {
    id: w.id,
    op,
    semantics: opSemantics(op),
    origin: (ORIGINS.includes(w.origin) ? w.origin : 'import') as Origin,
    concept: w.concept ?? null,
    addedAt: w.addedAt,
  };
}

function toItems(entries: unknown[]): BasketItem[] | null {
  const items: BasketItem[] = [];
  for (const entry of entries.slice(-MAX_ITEMS)) {
    const item = toItem(entry);
    if (item === null) return null;
    items.push(item);
  }
  return items;
}

/** A draft written by another schema version is refused; an unknown op type inside a v1 draft stays as an opaque item. */
export function decodeDraft(json: unknown): DecodedDraft {
  if (typeof json !== 'object' || json === null || Array.isArray(json)) return { ok: false, reason: 'invalid' };
  const version = (json as Record<string, unknown>)['schema_version'];
  if (typeof version !== 'number') return { ok: false, reason: 'invalid' };
  if (version !== DRAFT_SCHEMA_VERSION) return { ok: false, reason: 'schema_version' };
  const wire = WireDraft.safeParse(json);
  if (!wire.success) return { ok: false, reason: 'invalid' };
  const items = toItems(wire.data.basket);
  const shelf = toItems(wire.data.shelf);
  if (items === null || shelf === null) return { ok: false, reason: 'invalid' };
  return {
    ok: true,
    items,
    shelf,
    savedOps: wire.data.saved_ops.slice(-MAX_ITEMS),
    prompts: wire.data.prompts.slice(-MAX_PROMPTS).map((p) => clip(p, MAX_PROMPT_CHARS)),
    unsupported: items.filter((item) => isOpaque(item.op)).length,
  };
}
