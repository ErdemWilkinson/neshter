import { opKey, type KnownOp } from './ops';

/** Order-independent identity of an op set: sorted `opKey`s joined by `|`. */
export const opsFingerprint = (ops: readonly KnownOp[]): string =>
  ops
    .map(opKey)
    .sort()
    .join('|');
