export type CommentNodeData = {
  text?: string;
  url?: string;
};

export function defaultCommentData(): CommentNodeData {
  return { text: "", url: "" };
}
