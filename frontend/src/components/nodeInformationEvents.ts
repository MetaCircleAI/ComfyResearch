export const OPEN_NODE_INFORMATION_EVENT = "cr:open-node-information";

export type OpenNodeInformationDetail = {
  nodeId: string;
  title?: string;
  text?: string;
  code?: string;
  mode?: "parameters" | "code";
};

export function openNodeInformation(detail: OpenNodeInformationDetail): void {
  window.dispatchEvent(new CustomEvent<OpenNodeInformationDetail>(OPEN_NODE_INFORMATION_EVENT, { detail }));
}
