import { describe, expect, it } from "vitest";
import { defaultFourierComponentObservableData } from "./fourierComponentObservableDefaults";

describe("Fourier component observable defaults", () => {
  it("uses a stable scalar projection configuration", () => {
    expect(defaultFourierComponentObservableData()).toEqual({
      frequency: 1,
      metric: "relative_projection_mse",
      inputAxis: 0,
      outputIndex: 0,
    });
  });
});
