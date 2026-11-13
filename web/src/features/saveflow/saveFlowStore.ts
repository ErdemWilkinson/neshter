import { createStore } from 'zustand/vanilla';
import type { ErrorCode } from '@/api/errors';
import { IDLE, saveFlowReducer, type SaveFlowEvent, type SaveFlowState } from '@/core/saveflow';
import { basketStore } from '@/core/stores/basket';

export interface SaveFlowFailure {
  code: ErrorCode;
  detail: unknown;
  /** `export`: the save itself failed while its screen was gone; otherwise the re-run failed. */
  stage?: 'export';
}

export interface SaveFlowStoreState {
  state: SaveFlowState;
  /** Why the last re-run failed (shown once; cleared by the next `SAVE_CLICKED`, `CANCEL` or `DISMISS`). */
  error: SaveFlowFailure | null;
  dispatch(event: SaveFlowEvent): SaveFlowState;
  setError(error: SaveFlowFailure | null): void;
}

type ItemsSource = Pick<typeof basketStore, 'getState'>;

/** The pure reducer from core, fed with the live basket items. */
export const createSaveFlowStore = (basket: ItemsSource = basketStore) =>
  createStore<SaveFlowStoreState>()((set, get) => ({
    state: IDLE,
    error: null,
    dispatch: (event) => {
      const next = saveFlowReducer(get().state, event, { items: basket.getState().items });
      const clearsError = event.type === 'SAVE_CLICKED' || event.type === 'CANCEL' || event.type === 'DISMISS';
      set(clearsError ? { state: next, error: null } : { state: next });
      return next;
    },
    setError: (error) => set({ error }),
  }));

export const saveFlowStore = createSaveFlowStore();
