import type { SystemInfo } from '@/api/schemas';

export type FormatId = 'safetensors' | 'gguf_q4_k_m' | 'gguf_q8_0';
export type FormatNote = 'experimental' | 'needs_llama' | null;
export interface FormatOption {
  id: FormatId;
  enabled: boolean;
  note: FormatNote;
}

export const DEFAULT_FORMAT: FormatId = 'safetensors';

export const isGguf = (format: FormatId): boolean => format !== 'safetensors';

/** safetensors always; GGUF only with a llama.cpp that has its converter, and then marked experimental. */
export function formatOptions(system: Pick<SystemInfo, 'llama_cpp'> | null): FormatOption[] {
  const gguf = system !== null && system.llama_cpp.found && system.llama_cpp.converter;
  const note: FormatNote = gguf ? 'experimental' : 'needs_llama';
  return [
    { id: 'safetensors', enabled: true, note: null },
    { id: 'gguf_q4_k_m', enabled: gguf, note },
    { id: 'gguf_q8_0', enabled: gguf, note },
  ];
}

/** The Ollama toggle (default off) is possible only for GGUF and when `ollama` is on the PATH. */
export const ollamaAvailable = (system: Pick<SystemInfo, 'ollama'> | null, format: FormatId): boolean =>
  system !== null && system.ollama && isGguf(format);
