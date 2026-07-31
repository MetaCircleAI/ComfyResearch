/**
 * Compare current node spec metadata with a freshly parsed header to detect
 * renames or added/removed parameters (vs value-only edits).
 */

export function specStructureChanged(
  currentName: string,
  currentKeysSorted: string[],
  nextName: string,
  nextKeys: string[],
): boolean {
  if (currentName.trim() !== nextName.trim()) return true;
  const a = [...currentKeysSorted].sort().join("\0");
  const b = [...nextKeys].sort().join("\0");
  return a !== b;
}
