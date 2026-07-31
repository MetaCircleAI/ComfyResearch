export type TableVizNodeData = {
  plotXParamKey: string | null;
  logScaleX: boolean;
  logScaleY: boolean;
};

export function defaultTableVizData(): TableVizNodeData {
  return { plotXParamKey: null, logScaleX: false, logScaleY: false };
}
