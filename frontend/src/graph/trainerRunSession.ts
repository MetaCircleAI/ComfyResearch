/** In-flight trainer runs survive TrainerNode remount (project tab switch). */
const sessions = new Map<string, AbortController>();

export function registerTrainerRunSession(trainerId: string, ac: AbortController): void {
  sessions.set(trainerId, ac);
}

export function unregisterTrainerRunSession(trainerId: string): void {
  sessions.delete(trainerId);
}

export function getTrainerRunSession(trainerId: string): AbortController | null {
  return sessions.get(trainerId) ?? null;
}

export function hasTrainerRunSession(trainerId: string): boolean {
  return sessions.has(trainerId);
}
