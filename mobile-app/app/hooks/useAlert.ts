/**
 * app/hooks/useAlert.ts
 * ──────────────────────
 * Manages alert state — polls the /nearby API, triggers
 * vibration/sound, and exposes state to the Alert Screen.
 *
 * Usage:
 *   const { alertState, dismissAlert, isPolling } = useAlert(location);
 */

import { useCallback, useEffect, useRef, useState } from "react";
import * as Haptics from "expo-haptics";
import { AppState } from "react-native";
import { CONFIG } from "../constants/config";
import { getNearby, NearbyResponse } from "../services/api";
import { LocationCoords } from "./useLocation";

export interface AlertState {
  active:     boolean;
  level:      "HIGH" | "MEDIUM" | "SAFE";
  topScore:   number;
  reason:     string;
  response:   NearbyResponse | null;
  lastPolled: Date | null;
}

const INITIAL_STATE: AlertState = {
  active:     false,
  level:      "SAFE",
  topScore:   0,
  reason:     "",
  response:   null,
  lastPolled: null,
};

export function useAlert(location: LocationCoords | null) {
  const [alertState, setAlertState] = useState<AlertState>(INITIAL_STATE);
  const [isPolling,  setIsPolling]  = useState(false);
  const [apiError,   setApiError]   = useState<string | null>(null);
  const intervalRef  = useRef<ReturnType<typeof setInterval> | null>(null);
  const prevLevelRef = useRef<string>("SAFE");

  const poll = useCallback(async () => {
    if (!location) return;

    try {
      setIsPolling(true);
      const data = await getNearby(
        location.latitude,
        location.longitude,
        CONFIG.ALERT_RADIUS_M
      );

      const newLevel = data.alert_level;
      const wasAlert = prevLevelRef.current !== "SAFE";
      const isAlert  = newLevel !== "SAFE";

      // Trigger haptics only on new alerts (not repeated)
      if (isAlert && !wasAlert) {
        if (newLevel === "HIGH") {
          await Haptics.notificationAsync(
            Haptics.NotificationFeedbackType.Error
          );
        } else {
          await Haptics.notificationAsync(
            Haptics.NotificationFeedbackType.Warning
          );
        }
      }

      prevLevelRef.current = newLevel;

      setAlertState({
        active:     data.alert,
        level:      newLevel as AlertState["level"],
        topScore:   data.top_score,
        reason:     data.reason,
        response:   data,
        lastPolled: new Date(),
      });
      setApiError(null);
    } catch (err: any) {
      setApiError(err.message ?? "API error");
    } finally {
      setIsPolling(false);
    }
  }, [location]);

  // Poll on interval, pause when app is in background
  useEffect(() => {
    poll(); // immediate first call

    intervalRef.current = setInterval(poll, CONFIG.POLL_INTERVAL_MS);

    const sub = AppState.addEventListener("change", (state) => {
      if (state === "background" || state === "inactive") {
        if (intervalRef.current) clearInterval(intervalRef.current);
      } else if (state === "active") {
        poll();
        intervalRef.current = setInterval(poll, CONFIG.POLL_INTERVAL_MS);
      }
    });

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      sub.remove();
    };
  }, [poll]);

  const dismissAlert = useCallback(() => {
    prevLevelRef.current = "SAFE";
    setAlertState((prev) => ({ ...prev, active: false, level: "SAFE" }));
  }, []);

  return { alertState, dismissAlert, isPolling, apiError };
}