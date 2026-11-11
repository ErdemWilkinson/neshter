import type { ExportResult } from '@/api/schemas';
import { countSavable, type BasketItem } from './basket';
import { isSavable } from './ops';

export type SaveFlowState =
  | { kind: 'idle' }
  | { kind: 'nothingToSave' }
  | { kind: 'needsDiscard'; exploreIds: string[] }
  | { kind: 'recomparing'; dropping: string[]; jobId: string | null }
  | { kind: 'reviewReady'; dropped: number }
  | { kind: 'exportRequested' }
  | { kind: 'exporting'; jobId: string }
  | { kind: 'done'; result: ExportResult };

export type SaveFlowEvent =
  | { type: 'SAVE_CLICKED' }
  | { type: 'DISCARD_CONFIRMED' }
  | { type: 'RECOMPARE_STARTED'; jobId: string }
  | { type: 'RECOMPARE_DONE' }
  | { type: 'RECOMPARE_FAILED' }
  | { type: 'EXPORT_STARTED'; jobId: string }
  | { type: 'EXPORT_DONE'; result: ExportResult }
  | { type: 'EXPORT_FAILED' }
  | { type: 'CANCEL' }
  | { type: 'DISMISS' };

export const IDLE: SaveFlowState = { kind: 'idle' };

/** Pure transition function. Any (state, event) pair not listed returns `state` itself. */
export function saveFlowReducer(
  state: SaveFlowState,
  event: SaveFlowEvent,
  ctx: { items: readonly BasketItem[] },
): SaveFlowState {
  switch (event.type) {
    case 'SAVE_CLICKED': {
      if (state.kind !== 'idle' && state.kind !== 'reviewReady') return state;
      if (countSavable(ctx.items) === 0) return { kind: 'nothingToSave' };
      const exploreIds = ctx.items.filter((i) => !isSavable(i.op) && !i.pendingDiscard).map((i) => i.id);
      return exploreIds.length > 0 ? { kind: 'needsDiscard', exploreIds } : { kind: 'exportRequested' };
    }
    case 'DISCARD_CONFIRMED':
      return state.kind === 'needsDiscard'
        ? { kind: 'recomparing', dropping: state.exploreIds, jobId: null }
        : state;
    case 'RECOMPARE_STARTED':
      return state.kind === 'recomparing' ? { ...state, jobId: event.jobId } : state;
    case 'RECOMPARE_DONE':
      return state.kind === 'recomparing' ? { kind: 'reviewReady', dropped: state.dropping.length } : state;
    case 'RECOMPARE_FAILED':
    case 'CANCEL':
      return state.kind === 'done' || state.kind === 'idle' ? state : IDLE;
    case 'EXPORT_STARTED':
      return state.kind === 'exportRequested' ? { kind: 'exporting', jobId: event.jobId } : state;
    case 'EXPORT_DONE':
      return state.kind === 'exporting' ? { kind: 'done', result: event.result } : state;
    case 'EXPORT_FAILED':
      return state.kind === 'exporting' || state.kind === 'exportRequested' ? IDLE : state;
    case 'DISMISS':
      return state.kind === 'nothingToSave' || state.kind === 'done' || state.kind === 'reviewReady' ? IDLE : state;
  }
}
