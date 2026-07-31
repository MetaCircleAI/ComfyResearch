/** Default markdown for ModelNodeInfoModal / observable header (i). Keys match React Flow node `type` strings. */
import { nodeRegistryObservableInfoMarkdown } from "../../graph/nodeRegistrySpec";

const DEFAULT_INFO =
  "**Trainer observable** — connect this node’s observable output to the Trainer’s **observable** input and run **Train**. Pair an **Observable viz** node to plot streamed curves.";

export function getObservableInfoMarkdown(graphNodeType: string): string {
  return nodeRegistryObservableInfoMarkdown(graphNodeType) ?? DEFAULT_INFO;
}
