import type { ExportResult } from '@/api/schemas';
import type { BasketItem } from '../basket';
import { saveFlowReducer, type SaveFlowEvent, type SaveFlowState } from '../saveflow';
import { registerFakeExploreOp } from './fixtures/fakeExploreOp';
import { head, item } from './fixtures/items';

beforeAll(registerFakeExploreOp);

const fake = (extra?: Parameters<typeof item>[1]) => item({ type: 'test_explore', src: 'h:0:1' }, extra);
const savable = [head(0, 1), head(0, 2)];
const explore = fake();
const opaque = item({ type: 'edge_read_v9' });
const result: ExportResult = { path: 'C:\\x', source_sha256: 'a', source_untouched: true, safety_ablated: false, name: 'x' };

const idle: SaveFlowState = { kind: 'idle' };
const click: SaveFlowEvent = { type: 'SAVE_CLICKED' };

type Row = [string, SaveFlowState, SaveFlowEvent, BasketItem[], SaveFlowState];
const SAVEFLOW_TABLE: Row[] = [
  ['all savable -> exportRequested', idle, click, savable, { kind: 'exportRequested' }],
  ['savable + fake explore -> needsDiscard', idle, click, [...savable, explore], { kind: 'needsDiscard', exploreIds: [explore.id] }],
  ['only fake -> nothingToSave', idle, click, [explore], { kind: 'nothingToSave' }],
  ['empty basket -> nothingToSave', idle, click, [], { kind: 'nothingToSave' }],
  ['opaque counts as not savable', idle, click, [...savable, opaque], { kind: 'needsDiscard', exploreIds: [opaque.id] }],
  ['pending-discard items do not ask again', idle, click, [...savable, fake({ pendingDiscard: true })], { kind: 'exportRequested' }],
  ['DISCARD_CONFIRMED -> recomparing', { kind: 'needsDiscard', exploreIds: ['e1'] }, { type: 'DISCARD_CONFIRMED' }, [], { kind: 'recomparing', dropping: ['e1'], jobId: null }],
  ['RECOMPARE_STARTED sets the job id', { kind: 'recomparing', dropping: ['e1'], jobId: null }, { type: 'RECOMPARE_STARTED', jobId: 'j1' }, [], { kind: 'recomparing', dropping: ['e1'], jobId: 'j1' }],
  ['RECOMPARE_DONE -> reviewReady', { kind: 'recomparing', dropping: ['e1'], jobId: 'j1' }, { type: 'RECOMPARE_DONE' }, [], { kind: 'reviewReady', dropped: 1 }],
  ['CANCEL during recomparing -> idle', { kind: 'recomparing', dropping: ['e1'], jobId: 'j1' }, { type: 'CANCEL' }, [], idle],
  ['RECOMPARE_FAILED -> idle', { kind: 'recomparing', dropping: ['e1'], jobId: null }, { type: 'RECOMPARE_FAILED' }, [], idle],
  ['reviewReady + SAVE_CLICKED (now all savable) -> exportRequested', { kind: 'reviewReady', dropped: 1 }, click, savable, { kind: 'exportRequested' }],
  ['exportRequested + EXPORT_STARTED -> exporting', { kind: 'exportRequested' }, { type: 'EXPORT_STARTED', jobId: 'j2' }, [], { kind: 'exporting', jobId: 'j2' }],
  ['exporting + EXPORT_FAILED -> idle', { kind: 'exporting', jobId: 'j2' }, { type: 'EXPORT_FAILED' }, [], idle],
  ['exportRequested + EXPORT_FAILED -> idle', { kind: 'exportRequested' }, { type: 'EXPORT_FAILED' }, [], idle],
  ['exporting + EXPORT_DONE -> done', { kind: 'exporting', jobId: 'j2' }, { type: 'EXPORT_DONE', result }, [], { kind: 'done', result }],
  ['CANCEL while exporting -> idle', { kind: 'exporting', jobId: 'j2' }, { type: 'CANCEL' }, [], idle],
  ['DISMISS from nothingToSave -> idle', { kind: 'nothingToSave' }, { type: 'DISMISS' }, [], idle],
  ['DISMISS from done -> idle', { kind: 'done', result }, { type: 'DISMISS' }, [], idle],
  ['DISMISS from reviewReady -> idle', { kind: 'reviewReady', dropped: 2 }, { type: 'DISMISS' }, [], idle],
];

test.each(SAVEFLOW_TABLE)('saveflow %s', (_name, start, event, items, expected) => {
  expect(saveFlowReducer(start, event, { items })).toEqual(expected);
});

test.each<[string, SaveFlowState, SaveFlowEvent]>([
  ['done + CANCEL', { kind: 'done', result }, { type: 'CANCEL' }],
  ['done + RECOMPARE_FAILED', { kind: 'done', result }, { type: 'RECOMPARE_FAILED' }],
  ['idle + CANCEL', idle, { type: 'CANCEL' }],
  ['needsDiscard + SAVE_CLICKED', { kind: 'needsDiscard', exploreIds: ['e'] }, click],
  ['recomparing + SAVE_CLICKED', { kind: 'recomparing', dropping: [], jobId: null }, click],
  ['exporting + SAVE_CLICKED', { kind: 'exporting', jobId: 'j' }, click],
  ['idle + DISCARD_CONFIRMED', idle, { type: 'DISCARD_CONFIRMED' }],
  ['idle + RECOMPARE_DONE', idle, { type: 'RECOMPARE_DONE' }],
  ['idle + EXPORT_DONE', idle, { type: 'EXPORT_DONE', result }],
  ['exportRequested + EXPORT_DONE', { kind: 'exportRequested' }, { type: 'EXPORT_DONE', result }],
  ['needsDiscard + DISMISS', { kind: 'needsDiscard', exploreIds: ['e'] }, { type: 'DISMISS' }],
  ['idle + EXPORT_STARTED', idle, { type: 'EXPORT_STARTED', jobId: 'j' }],
])('saveflow %s returns the same state object', (_name, state, event) => {
  expect(saveFlowReducer(state, event, { items: savable })).toBe(state);
});
