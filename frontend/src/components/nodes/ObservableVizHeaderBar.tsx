import { useMemo } from "react";
import { useStore, type Node } from "@xyflow/react";
import { NodeSpecHeaderActions } from "./NodeSpecCodeFooter";
import { buildObservableNotebookStub } from "../../graph/observableNotebookStub";
import { getObservableInfoMarkdown } from "./observableNodeInfoMarkdown";

export function ObservableVizHeaderBar({
  id,
  pairedObservableId,
  title,
}: {
  id: string;
  pairedObservableId?: string;
  title: string;
}) {
  const pairedType = useStore((s) => {
    if (!pairedObservableId) return "observable_viz";
    const n = (s.nodes as Node[]).find((x) => x.id === pairedObservableId);
    return typeof n?.type === "string" && n.type ? n.type : "observable_viz";
  });

  const infoText = useMemo(() => {
    const pairedMd = getObservableInfoMarkdown(pairedType);
    return `**Observable viz (mirror)**\n\nPlots metrics streamed from the trainer for paired observable type \`${pairedType}\`.\n\n${pairedMd}`;
  }, [pairedType]);

  const generatedCode = useMemo(
    () => buildObservableNotebookStub("observable_viz", title, pairedType),
    [title, pairedType],
  );

  return (
    <div className="cr-node__header cr-node__header--dataset-info-row">
      <div className="cr-node__header-main">
        <span>{title}</span>
      </div>
      <div className="cr-dataset-node-header-actions">
        <NodeSpecHeaderActions
          nodeId={id}
          graphNodeType="observable_viz"
          generatedCode={generatedCode}
          infoTitle={title}
          infoText={infoText}
          codeKind="observable"
        />
      </div>
    </div>
  );
}
