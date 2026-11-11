import { useEffect } from 'react';
import { jobsStore } from '@/core/stores/jobs';

/** Call from any component that shows a running job's progress: the busy notice then stays quiet (it is redundant). */
export function useProgressShown(): void {
  useEffect(() => {
    jobsStore.getState().cardMounted();
    return () => jobsStore.getState().cardUnmounted();
  }, []);
}
