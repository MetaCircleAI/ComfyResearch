export type TensorViz1dPlotStyle = "line" | "scatter" | "bar" | "dist";

export type TensorViz1dLineSort = "original" | "descending" | "ascending";

/** Which axis holds fixed indices i1, i2 for scatter (other axis is point index). */
export type TensorViz2dScatterAxis = 0 | 1;

export type TensorViz2dPlotStyle = "heat" | "scatter";

export type TensorVizNodeData = {
  plot1dStyle: TensorViz1dPlotStyle;
  plot1dLineSort: TensorViz1dLineSort;
  histBins: number;
  plot2dStyle: TensorViz2dPlotStyle;
  plot2dScatterAxis: TensorViz2dScatterAxis;
  plot2dScatterI1: number;
  plot2dScatterI2: number;
};

export function defaultTensorVizData(): TensorVizNodeData {
  return {
    plot1dStyle: "line",
    plot1dLineSort: "original",
    histBins: 20,
    plot2dStyle: "scatter",
    plot2dScatterAxis: 1,
    plot2dScatterI1: 0,
    plot2dScatterI2: 1,
  };
}
