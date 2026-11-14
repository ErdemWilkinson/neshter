import { createStore } from 'zustand/vanilla';
import { subscribeWithSelector } from 'zustand/middleware';

export type Route = 'home' | 'models' | 'explore' | 'concept' | 'compare' | 'save';
export type View = 'hot' | '2d' | 'list' | '3d';
export type Lang = 'tr' | 'en';
export type ReducedMotion = 'system' | 'on' | 'off';
export type Quality = 'high' | 'balanced' | 'low';
/** What the person came to do (set by Home); `null` until they choose. */
export type Intent = 'reduce' | 'inspect' | 'expert' | null;

export const intentRoute = (intent: Intent): Route => (intent === 'reduce' ? 'concept' : 'explore');

export interface UiState {
  route: Route;
  view: View;
  lang: Lang;
  reducedMotion: ReducedMotion;
  quality: Quality;
  intent: Intent;
  setRoute(route: Route): void;
  setView(view: View): void;
  setLang(lang: Lang): void;
  setReducedMotion(reducedMotion: ReducedMotion): void;
  setQuality(quality: Quality): void;
  setIntent(intent: Intent): void;
}

export const createUiStore = () =>
  createStore<UiState>()(
    subscribeWithSelector((set) => ({
      route: 'home',
      view: '2d',
      lang: 'tr',
      reducedMotion: 'system',
      quality: 'balanced',
      intent: null,
      setRoute: (route) => set({ route }),
      setView: (view) => set({ view }),
      setLang: (lang) => set({ lang }),
      setReducedMotion: (reducedMotion) => set({ reducedMotion }),
      setQuality: (quality) => set({ quality }),
      setIntent: (intent) => set({ intent }),
    })),
  );

export const uiStore = createUiStore();
