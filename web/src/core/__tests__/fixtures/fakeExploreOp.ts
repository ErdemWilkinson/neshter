// Stand-in for a Plan 3 explore op: sendable to compare, but not savable.
import { registerOpKind } from '../../ops';
import { z } from '../../zod';

export const FAKE_EXPLORE_TYPE = 'test_explore';

export function registerFakeExploreOp(): void {
  registerOpKind(FAKE_EXPLORE_TYPE, {
    schema: z.strictObject({ type: z.literal(FAKE_EXPLORE_TYPE), src: z.string() }),
    savable: false,
    semantics: 'path',
  });
}
