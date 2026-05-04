/**
 * app/hooks/useLocation.ts
 * ─────────────────────────
 * Reusable hook that:
 *   1. Requests location permission on first use
 *   2. Watches GPS position with configurable interval
 *   3. Returns current coords, permission status, and error state
 *
 * Usage:
 *   const { location, hasPermission, error } = useLocation();
 */

import { useEffect, useRef, useState } from "react";
import * as Location from "expo-location";
import { CONFIG } from "../constants/config";

export interface LocationCoords {
  latitude:  number;
  longitude: number;
  accuracy?: number;
  heading?:  number;
  speed?:    number;
}

interface UseLocationReturn {
  location:      LocationCoords | null;
  hasPermission: boolean;
  isLoading:     boolean;
  error:         string | null;
}

export function useLocation(
  pollIntervalMs: number = CONFIG.POLL_INTERVAL_MS
): UseLocationReturn {
  const [location,      setLocation]      = useState<LocationCoords | null>(null);
  const [hasPermission, setHasPermission] = useState(false);
  const [isLoading,     setIsLoading]     = useState(true);
  const [error,         setError]         = useState<string | null>(null);
  const watcherRef = useRef<Location.LocationSubscription | null>(null);

  useEffect(() => {
    let mounted = true;

    async function startWatching() {
      try {
        // Request permission
        const { status } = await Location.requestForegroundPermissionsAsync();
        if (!mounted) return;

        if (status !== "granted") {
          setHasPermission(false);
          setError("Location permission denied. Please enable it in Settings.");
          setIsLoading(false);
          return;
        }

        setHasPermission(true);

        // Get immediate position first
        const current = await Location.getCurrentPositionAsync({
          accuracy: Location.Accuracy.Balanced,
        });
        if (mounted) {
          setLocation({
            latitude:  current.coords.latitude,
            longitude: current.coords.longitude,
            accuracy:  current.coords.accuracy ?? undefined,
            heading:   current.coords.heading ?? undefined,
            speed:     current.coords.speed ?? undefined,
          });
          setIsLoading(false);
        }

        // Then watch for updates
        watcherRef.current = await Location.watchPositionAsync(
          {
            accuracy:         Location.Accuracy.Balanced,
            timeInterval:     pollIntervalMs,
            distanceInterval: 10,   // update only after moving 10m
          },
          (loc) => {
            if (mounted) {
              setLocation({
                latitude:  loc.coords.latitude,
                longitude: loc.coords.longitude,
                accuracy:  loc.coords.accuracy ?? undefined,
                heading:   loc.coords.heading ?? undefined,
                speed:     loc.coords.speed ?? undefined,
              });
            }
          }
        );
      } catch (err: any) {
        if (mounted) {
          setError(err.message ?? "Failed to get location");
          setIsLoading(false);
        }
      }
    }

    startWatching();

    return () => {
      mounted = false;
      watcherRef.current?.remove();
    };
  }, [pollIntervalMs]);

  return { location, hasPermission, isLoading, error };
}