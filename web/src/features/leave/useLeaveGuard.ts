import { createElement, useCallback, useEffect, useState, type ReactNode } from 'react';
import { useStore } from 'zustand';
import { navigate } from '@/app/router';
import { isDirty } from '@/core/basket';
import { basketStore } from '@/core/stores/basket';
import { LeaveDialog } from './LeaveDialog';

export type LeaveKind = 'switch_model' | 'close' | 'home';

interface Pending {
  kind: LeaveKind;
  proceed: () => void;
}

/**
 * `request(kind, proceed)` runs `proceed` at once when nothing is unsaved; otherwise it asks first. `Kaydet` goes to
 * the Save screen without proceeding, `Kaydetmeden çık` proceeds (autosave keeps the server draft, so the work can
 * still be restored), `Vazgeç` changes nothing. While dirty, closing or reloading the tab triggers the browser's
 * own `beforeunload` confirmation (unless `beforeUnload: false`).
 */
export function useLeaveGuard(
  { beforeUnload = true }: { beforeUnload?: boolean } = {},
): { request(kind: LeaveKind, proceed: () => void): void; dialog: ReactNode } {
  const dirty = useStore(basketStore, (s) => isDirty(s.items, s.savedOps));
  const [pending, setPending] = useState<Pending | null>(null);

  // Only one mount arms the tab-close confirmation (App); it is off once the server is shut down.
  useEffect(() => {
    if (!dirty || !beforeUnload) return;
    const onBeforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = '';
    };
    window.addEventListener('beforeunload', onBeforeUnload);
    return () => window.removeEventListener('beforeunload', onBeforeUnload);
  }, [dirty, beforeUnload]);

  const request = useCallback((kind: LeaveKind, proceed: () => void) => {
    const { items, savedOps } = basketStore.getState();
    if (isDirty(items, savedOps)) setPending({ kind, proceed });
    else proceed();
  }, []);

  const dialog = createElement(LeaveDialog, {
    open: pending !== null,
    onSave: () => {
      setPending(null);
      navigate('save');
    },
    onLeave: () => {
      const leaving = pending;
      setPending(null);
      leaving?.proceed();
    },
    onCancel: () => setPending(null),
  });

  return { request, dialog };
}
