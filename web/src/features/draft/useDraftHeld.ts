import { useStore } from 'zustand';
import { sessionStore } from '@/core/stores/session';
import { draftStatusStore } from './draftStatus';

/**
 * Autosave is held back for the loaded model (an earlier draft is still unresolved, `pendingSha`): cuts made now are
 * not written to the draft, so leave/closed/stale copy must not promise they are.
 */
export function useDraftHeld(): boolean {
  const sha = useStore(sessionStore, (s) => s.model?.source_sha256 ?? null);
  const pending = useStore(draftStatusStore, (s) => s.pendingSha);
  return pending !== null && pending === sha;
}
