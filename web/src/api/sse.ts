import { ApiError, apiErrorFromResponse, isAbortError } from './errors';

export interface SseOptions {
  token: string;
  signal: AbortSignal;
  onEvent: (event: string, data: unknown) => void;
  fetchFn?: typeof fetch;
}

function parseData(raw: string): unknown {
  try {
    return JSON.parse(raw);
  } catch {
    return raw;
  }
}

/** Incremental `text/event-stream` parser: LF, CRLF and lone CR line ends, `:` comments, multi-line data. */
function createParser(onEvent: SseOptions['onEvent']) {
  let buffer = '';
  let event = '';
  let data: string[] = [];

  const dispatch = () => {
    if (data.length > 0) onEvent(event || 'message', parseData(data.join('\n')));
    event = '';
    data = [];
  };

  const line = (text: string) => {
    if (text === '') return dispatch();
    if (text.startsWith(':')) return;
    const colon = text.indexOf(':');
    const field = colon < 0 ? text : text.slice(0, colon);
    let value = colon < 0 ? '' : text.slice(colon + 1);
    if (value.startsWith(' ')) value = value.slice(1);
    if (field === 'event') event = value;
    else if (field === 'data') data.push(value);
  };

  return {
    push(chunk: string) {
      buffer += chunk;
      const lineEnd = /\r\n|\r|\n/g;
      let start = 0;
      for (let m = lineEnd.exec(buffer); m !== null; m = lineEnd.exec(buffer)) {
        // A trailing CR may be the first half of a CRLF split across chunks: wait for the next chunk.
        if (m[0] === '\r' && lineEnd.lastIndex === buffer.length) break;
        line(buffer.slice(start, m.index));
        start = lineEnd.lastIndex;
      }
      buffer = buffer.slice(start);
    },
  };
}

/**
 * Reads a server-sent-event stream with `fetch` (EventSource cannot send the token header). Resolves `'ended'`
 * when the server closes the stream or the connection drops, `'aborted'` when `signal` fires. A non-OK response
 * throws the mapped `ApiError`; a failed connection throws `ApiError(0, 'network')`.
 */
export async function streamSse(url: string, opts: SseOptions): Promise<'ended' | 'aborted'> {
  const { token, signal, onEvent } = opts;
  const fetchFn: typeof fetch = opts.fetchFn ?? ((input, init) => fetch(input, init));

  let res: Response;
  try {
    res = await fetchFn(url, { headers: { 'X-Neshter-Token': token, Accept: 'text/event-stream' }, signal });
  } catch (error) {
    if (signal.aborted || isAbortError(error)) return 'aborted';
    if (error instanceof TypeError) throw new ApiError(0, 'network', error.message);
    throw error;
  }
  if (!res.ok) throw await apiErrorFromResponse(res);
  if (!res.body) return 'ended';

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  const parser = createParser(onEvent);
  try {
    for (;;) {
      // Only the read is guarded: an error thrown by `onEvent` (consumer code) must surface, not look like a drop.
      let chunk: ReadableStreamReadResult<Uint8Array>;
      try {
        chunk = await reader.read();
      } catch (error) {
        if (signal.aborted || isAbortError(error)) return 'aborted';
        if (error instanceof TypeError) return 'ended'; // connection reset mid-stream: the caller re-attaches
        throw error;
      }
      if (chunk.done) break;
      parser.push(decoder.decode(chunk.value, { stream: true }));
    }
    parser.push(decoder.decode());
  } finally {
    reader.cancel().catch(() => undefined);
  }
  return signal.aborted ? 'aborted' : 'ended';
}
