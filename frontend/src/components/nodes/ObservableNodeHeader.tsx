import { useMemo } from "react";
import { NodeSpecHeaderActions } from "./NodeSpecCodeFooter";
import { buildObservableNotebookStub } from "../../graph/observableNotebookStub";
import { getObservableInfoMarkdown } from "./observableNodeInfoMarkdown";

export function ObservableNodeHeader({
  id,
  graphNodeType,
  title,
}: {
  id: string;
  graphNodeType: string;
  title: string;
}) {
  const infoText = useMemo(() => getObservableInfoMarkdown(graphNodeType), [graphNodeType]);
  const generatedCode = useMemo(
    () => buildObservableNotebookStub(graphNodeType, title),
    [graphNodeType, title],
  );
  return (
    <div className="cr-node__header cr-node__header--dataset-info-row">
      <div className="cr-node__header-main">
        <span>{title}</span>
      </div>
      <div className="cr-dataset-node-header-actions">
        <NodeSpecHeaderActions
          nodeId={id}
          graphNodeType={graphNodeType}
          generatedCode={generatedCode}
          infoTitle={title}
          infoText={infoText}
          codeKind="observable"
        />
      </div>
    </div>
  );
}
