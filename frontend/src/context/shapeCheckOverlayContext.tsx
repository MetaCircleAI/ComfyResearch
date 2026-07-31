import { createContext, useContext, useMemo, useState, type ReactNode } from "react";

export type ShapeCheckOverlayApi = {
  errorNodeIds: ReadonlySet<string>;
  setShapeCheckErrors: (ids: string[]) => void;
  clearShapeCheckErrors: () => void;
};

const ShapeCheckOverlayContext = createContext<ShapeCheckOverlayApi | null>(null);

export function ShapeCheckOverlayProvider({ children }: { children: ReactNode }) {
  const [errors, setErrors] = useState<string[]>([]);
  const value = useMemo<ShapeCheckOverlayApi>(
    () => ({
      errorNodeIds: new Set(errors),
      setShapeCheckErrors: (ids) => setErrors([...new Set(ids)]),
      clearShapeCheckErrors: () => setErrors([]),
    }),
    [errors],
  );
  return <ShapeCheckOverlayContext.Provider value={value}>{children}</ShapeCheckOverlayContext.Provider>;
}

export function useShapeCheckOverlay(): ShapeCheckOverlayApi | null {
  return useContext(ShapeCheckOverlayContext);
}
