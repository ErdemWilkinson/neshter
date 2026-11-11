/** Seconds left, extrapolated from elapsed time; `null` until 5 % progress so no invented estimate is shown. */
export function etaSeconds(progress: number, elapsedMs: number): number | null {
  if (!(progress >= 0.05)) return null;
  return (elapsedMs * (1 - progress)) / progress / 1000;
}
