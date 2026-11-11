export const ERROR_CODES = [
  'forbidden',
  'not_found',
  'busy',
  'invalid_input',
  'no_model',
  'oom',
  'unsupported_arch',
  'unsafe_file',
  'llama_cpp_missing',
  'ollama_missing',
  'disk_full',
  'source_modified',
  'exists',
  'cancelled',
  'internal',
  'too_large',
  'unsupported',
  'network',
] as const;

export type ErrorCode = (typeof ERROR_CODES)[number];

export function toErrorCode(value: unknown): ErrorCode | null {
  return typeof value === 'string' && (ERROR_CODES as readonly string[]).includes(value) ? (value as ErrorCode) : null;
}

/** `detail` is server text (attacker-influenced): show it only inside the "Teknik ayrıntı" disclosure, as text. */
export class ApiError extends Error {
  readonly status: number;
  readonly code: ErrorCode;
  readonly detail: unknown;

  constructor(status: number, code: ErrorCode, detail?: unknown) {
    super(`${code} (${status})`);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.detail = detail;
  }
}

export class BusyError extends ApiError {
  constructor(detail?: unknown) {
    super(409, 'busy', detail);
    this.name = 'BusyError';
  }
}

export function isAbortError(error: unknown): boolean {
  return typeof error === 'object' && error !== null && (error as { name?: unknown }).name === 'AbortError';
}

const STATUS_CODES: Record<number, ErrorCode> = {
  403: 'forbidden',
  404: 'not_found',
  409: 'busy',
  413: 'too_large',
  422: 'invalid_input',
  501: 'unsupported',
};

/** Maps a non-OK response: the server's `error_code` wins, then the status, then `internal`. */
export async function apiErrorFromResponse(res: Response): Promise<ApiError> {
  const text = await res.text().catch(() => '');
  let body: unknown;
  try {
    body = JSON.parse(text);
  } catch {
    body = undefined;
  }
  const record = typeof body === 'object' && body !== null ? (body as Record<string, unknown>) : {};
  const code = toErrorCode(record['error_code']) ?? STATUS_CODES[res.status] ?? 'internal';
  const detail = 'detail' in record ? record['detail'] : text;
  return res.status === 409 && code === 'busy' ? new BusyError(detail) : new ApiError(res.status, code, detail);
}
