// The phone app's bridge: the Android WebView exposes `window.NeshterNative` (a JavascriptInterface). A call is
// fire-and-forget on the native side; the answer comes back later through `window.__neshterReply`, so the model
// work never blocks the page. Payloads are JSON in both directions. Long calls (generation, copying a model in)
// also send `window.__neshterEvent(id, json)` while they run.

export interface NativeHost {
  call(id: string, method: string, argsJson: string): void;
}

export interface Bridge {
  /** onEvent sees the call's progress events (streamed text, bytes copied) until the answer arrives. */
  invoke<T>(method: string, args?: unknown, onEvent?: (payload: unknown) => void): Promise<T>;
}

declare global {
  interface Window {
    NeshterNative?: NativeHost;
    __neshterReply?: (id: string, ok: boolean, payloadJson: string) => void;
    __neshterEvent?: (id: string, payloadJson: string) => void;
  }
}

/** True inside the Android app (the native host is injected before the page loads). */
export const isDeviceApp = (): boolean => typeof window !== 'undefined' && window.NeshterNative !== undefined;

export class NativeError extends Error {
  constructor(
    message: string,
    readonly code: string,
  ) {
    super(message);
  }
}

export function createBridge(host: NativeHost, target: Window = window): Bridge {
  const pending = new Map<
    string,
    { resolve: (v: unknown) => void; reject: (e: unknown) => void; onEvent?: ((payload: unknown) => void) | undefined }
  >();
  let next = 0;
  target.__neshterEvent = (id, payloadJson) => {
    const p = pending.get(id);
    if (p?.onEvent === undefined) return;
    try {
      p.onEvent(JSON.parse(payloadJson));
    } catch {
      // a malformed or throwing progress event never breaks the call itself
    }
  };
  target.__neshterReply = (id, ok, payloadJson) => {
    const p = pending.get(id);
    if (p === undefined) return;
    pending.delete(id);
    let payload: unknown;
    try {
      payload = JSON.parse(payloadJson);
    } catch {
      p.reject(new NativeError(`Bad native reply: ${payloadJson.slice(0, 120)}`, 'internal'));
      return;
    }
    if (ok) p.resolve(payload);
    else {
      const e = (payload ?? {}) as { message?: string; code?: string };
      p.reject(new NativeError(e.message ?? 'native error', e.code ?? 'internal'));
    }
  };
  return {
    invoke<T>(method: string, args: unknown = {}, onEvent?: (payload: unknown) => void) {
      const id = `c${++next}`;
      return new Promise<T>((resolve, reject) => {
        pending.set(id, { resolve: resolve as (v: unknown) => void, reject, onEvent });
        try {
          host.call(id, method, JSON.stringify(args));
        } catch (error) {
          pending.delete(id);
          reject(error);
        }
      });
    },
  };
}
