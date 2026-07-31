export type CrlEnvPreset = "point_u4_maze" | "ant_u4_maze";

export type CrlEnvConfigNodeData = {
  preset: CrlEnvPreset;
  numEnvs: number;
  episodeLength: number;
  mazeSizeScaling: number;
  seed: number;
};

export function defaultCrlEnvConfigData(): CrlEnvConfigNodeData {
  return {
    preset: "point_u4_maze",
    numEnvs: 8,
    episodeLength: 200,
    mazeSizeScaling: 4,
    seed: 0,
  };
}
