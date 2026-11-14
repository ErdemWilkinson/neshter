import { useEffect, useRef, useState } from 'react';
import { basketStore } from '@/core/stores/basket';
import { plural } from '@/i18n';

const SHOW_MS = 2800;

/**
 * A short visual note whenever something is added to the basket (from the map, the hunt or the list; not restores), so a click on "add" is seen
 * to land. It is decoration: the basket panels and the header button already say it in words for screen readers.
 */
export function BasketToast() {
  const [shown, setShown] = useState<{ added: number; total: number; key: number } | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const unsubscribe = basketStore.subscribe(
      (s) => s.lastAdded,
      (last) => {
        if (last === null) return;
        const total = basketStore.getState().items.length;
        setShown((prev) => ({ added: last.n, total, key: (prev?.key ?? 0) + 1 }));
        if (timer.current !== null) clearTimeout(timer.current);
        timer.current = setTimeout(() => setShown(null), SHOW_MS);
      },
    );
    return () => {
      unsubscribe();
      if (timer.current !== null) clearTimeout(timer.current);
    };
  }, []);

  if (shown === null) return null;
  return (
    <div key={shown.key} className="basket-toast" aria-hidden="true">
      <b>{plural('toast.added', shown.added)}</b> <span>{plural('toast.total', shown.total)}</span>
    </div>
  );
}
