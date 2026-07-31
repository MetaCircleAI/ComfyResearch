import { createContext } from "react";
import type { Connection } from "@xyflow/react";

export type ConnectionSnapTarget = {
  connection: Connection;
  x: number;
  y: number;
};

export const ConnectionSnapContext = createContext<ConnectionSnapTarget | null>(null);
