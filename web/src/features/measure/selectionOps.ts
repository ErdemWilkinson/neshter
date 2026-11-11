import { parseNodeId, type NodeId } from '@/core/ids';
import type { HeadOp } from '@/core/ops';

/** The selected heads as ops, sorted by layer then head (other node kinds are not measurable here). */
export function selectedHeadOps(selected: ReadonlySet<NodeId>): HeadOp[] {
  const ops: HeadOp[] = [];
  for (const id of selected) {
    const node = parseNodeId(id);
    if (node?.kind === 'head') ops.push({ type: 'head', layer: node.layer, head: node.head });
  }
  return ops.sort((a, b) => a.layer - b.layer || a.head - b.head);
}
