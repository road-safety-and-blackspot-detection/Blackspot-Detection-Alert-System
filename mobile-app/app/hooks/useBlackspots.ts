/**
 * app/hooks/useBlackspots.ts
 * ───────────────────────────
 * Fetches and caches all blackspot data for the Map Screen.
 * Refetches only when explicitly called — not on every render.
 *
 * Usage:
 *   const { blackspots, isLoading, error, refetch } = useBlackspots();
 */

import { useCallback, useEffect, useState } from "react";
import { BlackSpot, getAllBlackspots } from "../services/api";

interface UseBlackspotsReturn {
  blackspots: BlackSpot[];
  isLoading:  boolean;
  error:      string | null;
  refetch:    () => Promise<void>;
  lastFetched: Date | null;
}

export function useBlackspots(
  riskLevel?: "HIGH" | "MEDIUM" | "LOW"
): UseBlackspotsReturn {
  const [blackspots,  setBlackspots]  = useState<BlackSpot[]>([]);
  const [isLoading,   setIsLoading]   = useState(true);
  const [error,       setError]       = useState<string | null>(null);
  const [lastFetched, setLastFetched] = useState<Date | null>(null);

  const fetch = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      const data = await getAllBlackspots(riskLevel);
      setBlackspots(data.blackspots ?? []);
      setLastFetched(new Date());
    } catch (err: any) {
      setError(err.message ?? "Failed to load blackspots");
    } finally {
      setIsLoading(false);
    }
  }, [riskLevel]);

  useEffect(() => {
    fetch();
  }, [fetch]);

  return {
    blackspots,
    isLoading,
    error,
    refetch:     fetch,
    lastFetched,
  };
}