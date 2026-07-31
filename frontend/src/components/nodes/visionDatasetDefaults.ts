import type { ListOr1 } from "./multiValueUtils";

export const VISION_DATASET_KINDS = [
  "mnist_dataset",
  "cifar10_dataset",
  "gaussian_blob_dataset",
  "shape_world_dataset",
  "hole_counting_dataset",
] as const;
export type VisionDatasetKind = (typeof VISION_DATASET_KINDS)[number];

export type CifarInputTransform = "none" | "shuffled_pixels" | "random_pixels" | "gaussian";
export const CIFAR_INPUT_TRANSFORM_OPTIONS: readonly CifarInputTransform[] = [
  "none",
  "shuffled_pixels",
  "random_pixels",
  "gaussian",
];
export type CifarPreprocessing = "none" | "center_crop_28_per_image_whiten";
export const CIFAR_PREPROCESSING_OPTIONS: readonly CifarPreprocessing[] = [
  "none",
  "center_crop_28_per_image_whiten",
];

export type VisionDatasetNodeData = {
  trainSize: ListOr1<number>;
  testSize: ListOr1<number>;
  initSeed: ListOr1<number>;
  seed: ListOr1<number>;
  /** Reshape [N,1,H,W] → [N,H·W] for MLP + cross_entropy_loss (disable for ResNet/ViT). */
  flattenOutput?: boolean;
  samplingMode?: "fixed" | "streaming";
  /** MNIST: optional cache directory for IDX downloads (empty = default under ~/.cache). */
  downloadCacheDir?: string;
  /** gaussian_blob_dataset, shape_world_dataset, hole_counting_dataset */
  imageSize?: ListOr1<number>;
  /** gaussian_blob_dataset: Gaussian std on each sampled image (prototype jitter scales with this).
   *  shape_world_dataset: additive N(0,1) noise scale on the image (0 = clean shapes on flat bg). */
  noiseLevel?: ListOr1<number>;
  /** gaussian_blob_dataset: number of classes (labels 0 … numClasses − 1). */
  numClasses?: ListOr1<number>;
  /** hole_counting_dataset only */
  maxHoles?: ListOr1<number>;
  /** CIFAR-10 reproducibility subset configuration. */
  subsetSeed?: ListOr1<number>;
  classBalanced?: boolean;
  normalize?: "zero_one" | "minus_one_to_one";
  /** CIFAR-10 only: fixed input randomization used by Zhang et al. Figure 1(a). */
  inputTransform?: CifarInputTransform;
  /** CIFAR-10 only: deterministic image preprocessing after any fixed randomization. */
  preprocessing?: CifarPreprocessing;
  /** CIFAR-10 only: fraction of training labels replaced once and then held fixed. */
  labelCorruption?: ListOr1<number>;
  specCodeName?: string;
  paramOrder?: string[];
  extras?: Record<string, unknown>;
};

export function defaultVisionDatasetData(kind: VisionDatasetKind): VisionDatasetNodeData {
  const base: VisionDatasetNodeData = {
    trainSize: 2048,
    testSize: 512,
    initSeed: 0,
    seed: 0,
    flattenOutput: false,
    samplingMode: "fixed",
    specCodeName: `${kind}Spec`,
    paramOrder: defaultParamOrderFor(kind),
  };
  if (kind === "mnist_dataset") {
    return {
      ...base,
      downloadCacheDir: "",
    };
  }
  if (kind === "cifar10_dataset") {
    return {
      ...base,
      downloadCacheDir: "",
      subsetSeed: 0,
      classBalanced: true,
      normalize: "zero_one",
      inputTransform: "none",
      preprocessing: "none",
      labelCorruption: 0,
    };
  }
  if (kind === "gaussian_blob_dataset") {
    return {
      ...base,
      numClasses: 10,
      imageSize: 28,
      noiseLevel: 0.15,
    };
  }
  if (kind === "shape_world_dataset") {
    return { ...base, imageSize: 32, noiseLevel: 0.04 };
  }
  return { ...base, imageSize: 48, maxHoles: 3 };
}

export function defaultParamOrderFor(kind: VisionDatasetKind): string[] {
  if (kind === "cifar10_dataset") {
    return [
      "downloadCacheDir",
      "subsetSeed",
      "classBalanced",
      "inputTransform",
      "preprocessing",
      "labelCorruption",
      "normalize",
      "flattenOutput",
      "samplingMode",
      "trainSize",
      "testSize",
      "initSeed",
    ];
  }
  if (kind === "mnist_dataset") {
    return ["downloadCacheDir", "flattenOutput", "samplingMode", "trainSize", "testSize", "initSeed"];
  }
  if (kind === "gaussian_blob_dataset") {
    return [
      "numClasses",
      "imageSize",
      "noiseLevel",
      "flattenOutput",
      "samplingMode",
      "trainSize",
      "testSize",
      "initSeed",
    ];
  }
  if (kind === "shape_world_dataset") {
    return [
      "imageSize",
      "noiseLevel",
      "flattenOutput",
      "samplingMode",
      "trainSize",
      "testSize",
      "initSeed",
    ];
  }
  return ["imageSize", "maxHoles", "flattenOutput", "samplingMode", "trainSize", "testSize", "initSeed"];
}
