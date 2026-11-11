import { useEffect, useId, useRef, type ReactNode } from 'react';

const FOCUSABLE = 'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])';

/**
 * A native modal `<dialog>`. Esc (the browser's `cancel` event) calls `onCancel` and leaves closing to the parent
 * through `open`. Initial focus goes to the first action.
 */
export function Dialog({
  open,
  title,
  children,
  actions,
  onCancel,
}: {
  open: boolean;
  title: string;
  children?: ReactNode;
  actions: ReactNode;
  onCancel: () => void;
}) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const actionsRef = useRef<HTMLDivElement>(null);
  const titleId = useId();

  useEffect(() => {
    const dialog = dialogRef.current;
    if (dialog === null) return;
    if (open && !dialog.open) {
      // jsdom and very old browsers lack showModal; the attribute still shows the dialog.
      if (typeof dialog.showModal === 'function') dialog.showModal();
      else dialog.setAttribute('open', '');
      actionsRef.current?.querySelector<HTMLElement>(FOCUSABLE)?.focus();
    } else if (!open && dialog.open) {
      if (typeof dialog.close === 'function') dialog.close();
      else dialog.removeAttribute('open');
    }
  }, [open]);

  return (
    <dialog
      ref={dialogRef}
      className="dialog glass"
      aria-labelledby={titleId}
      onCancel={(event) => {
        event.preventDefault(); // the parent decides whether the dialog closes
        onCancel();
      }}
    >
      <h2 id={titleId}>{title}</h2>
      {children}
      <div className="dialog-actions" ref={actionsRef}>
        {actions}
      </div>
    </dialog>
  );
}
