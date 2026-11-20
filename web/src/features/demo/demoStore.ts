import { createStore } from 'zustand/vanilla';

/** The one-click example: the sample model reads this text and the strongest heads at its last token are measured. */
export const DEMO_PROMPT = 'The capital of France is';
export const DEMO_HEADS = 2;

/**
 * Where the example is: `load` until Models starts the sample download, `run` until Explore sends the text, `pick`
 * until the map arrives and the heads are chosen, `measure` until the measure button starts. `idle` is no example.
 */
export type DemoStep = 'idle' | 'load' | 'run' | 'pick' | 'measure';

export interface DemoState {
  step: DemoStep;
  /** The explanation stays on Explore after the steps are done, until the person types their own text. */
  shown: boolean;
  setStep(step: DemoStep): void;
  start(): void;
  stop(): void;
}

export const createDemoStore = () =>
  createStore<DemoState>()((set) => ({
    step: 'idle',
    shown: false,
    setStep: (step) => set({ step }),
    start: () => set({ step: 'load', shown: true }),
    stop: () => set({ step: 'idle', shown: false }),
  }));

export const demoStore = createDemoStore();
