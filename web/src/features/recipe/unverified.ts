import type { BasketItem } from '@/core/basket';
import { isDirectionOp } from '@/core/ops';

/**
 * A direction vector that came from an imported recipe: nobody here checked that it means what its file says
 * (F30). Task 11's `BasketRow` reuses this so the badge stays on the item once it is in the basket.
 */
export const isUnverifiedDirection = (item: BasketItem): boolean => item.origin === 'import' && isDirectionOp(item.op);
