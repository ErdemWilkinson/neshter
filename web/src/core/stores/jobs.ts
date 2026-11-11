import { createStore } from 'zustand/vanilla';
import { subscribeWithSelector } from 'zustand/middleware';
import type { JobState } from '@/api/schemas';

export interface JobsState {
  /** The job that holds the model, while it is queued or running. */
  active: JobState<unknown> | null;
  setActive(job: JobState<unknown>): void;
  /** Clears the active job; with `id`, only when it is that job (a newer job is left alone). */
  clearActive(id?: string): void;
  /** How many `ProgressCard`s are on screen: a running job that a card already shows needs no extra banner. */
  progressCards: number;
  cardMounted(): void;
  cardUnmounted(): void;
}

export const createJobsStore = () =>
  createStore<JobsState>()(
    subscribeWithSelector((set) => ({
      active: null,
      setActive: (job) => set({ active: job }),
      clearActive: (id) => set((s) => (id === undefined || s.active?.id === id ? { active: null } : s)),
      progressCards: 0,
      cardMounted: () => set((s) => ({ progressCards: s.progressCards + 1 })),
      cardUnmounted: () => set((s) => ({ progressCards: Math.max(0, s.progressCards - 1) })),
    })),
  );

export const jobsStore = createJobsStore();

export const isLocked = (s: Pick<JobsState, 'active'>): boolean =>
  s.active !== null && (s.active.state === 'queued' || s.active.state === 'running');
