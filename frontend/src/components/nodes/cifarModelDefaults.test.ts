import { describe, expect, it } from "vitest";
import { GENERATED_NODE_SPECS } from "../../generated/generatedNodeSpecs";
import {
  defaultSmallInceptionCifarModelData,
  defaultVgg11CifarModelData,
} from "./cifarModelDefaults";

describe("CIFAR model defaults", () => {
  it("uses a distinct Small Inception spec name without changing VGG defaults", () => {
    expect(defaultSmallInceptionCifarModelData()).toEqual({ seed: 0, specCodeName: "smallInceptionCifarModelSpec" });
    expect(defaultVgg11CifarModelData()).toEqual({ seed: 0, specCodeName: "vgg11CifarModelSpec" });
    expect(GENERATED_NODE_SPECS.small_inception_cifar_model).toMatchObject({
      label: "Small Inception (CIFAR)",
      defaults: { seed: 0, specCodeName: "smallInceptionCifarModelSpec" },
      frontend: { componentKey: "Vgg11CifarModelNode" },
    });
  });
});
