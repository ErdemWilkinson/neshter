import { useEffect, useState } from 'react';

type Listener = (text: string) => void;
const listeners = new Set<Listener>();

/** Speaks `text` through the mounted `<LiveRegion />` (polite: it waits for the screen reader to finish). */
export function announce(text: string): void {
  for (const listener of listeners) listener(text);
}

export function LiveRegion() {
  const [message, setMessage] = useState({ text: '', id: 0 });
  useEffect(() => {
    const listener: Listener = (text) => setMessage((previous) => ({ text, id: previous.id + 1 }));
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  }, []);
  return (
    <div className="sr-only" aria-live="polite" aria-atomic="true" role="status">
      {/* A new element per message makes the same text announce again. */}
      <span key={message.id}>{message.text}</span>
    </div>
  );
}
