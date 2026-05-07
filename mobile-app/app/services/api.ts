/**
 * app/services/api.ts
 * ────────────────────
 * Single source of truth for ALL API calls.
 * Every screen imports from here — never write fetch() directly in screens.
 *
 * If the API URL changes, or you add auth headers, you change it here only.
 */

// import { CONFIG } from "../constants/config";

//  ── Types (mirror backend Pydantic models) ──

// export interface BlackSpot {
//   cluster_id:           number;
//   lat:                  number;
//   lng:                  number;
//   accident_count:       number;
//   total_killed:         number;
//   avg_severity:         number;
//   avg_env_risk:         number;
//   state_weight:         number;
//   night_accident_pct:   number;
//   monsoon_accident_pct: number;
//   primary_road_type:    string;
//   top_weather:          string;
//   risk_score:           number;
//   risk_level:           "HIGH" | "MEDIUM" | "LOW";
//   distance_m?:          number;
// }

// export interface NearbyResponse {
//   alert:       boolean;
//   alert_level: "HIGH" | "MEDIUM" | "SAFE";
//   top_score:   number;
//   black_spots: BlackSpot[];
//   reason:      string;
//   queried_at?: string;
// }

// export interface RouteRiskResponse {
//   overall_risk_score:  number;
//   risk_level:          string;
//   total_blackspots:    number;
//   high_risk_zones:     number;
//   blackspots_on_route: BlackSpot[];
// }

// export interface RouteCompareResponse {
//   route_a:         RouteRiskResponse;
//   route_b:         RouteRiskResponse;
//   safer_route:     "A" | "B";
//   risk_difference: number;
//   recommendation:  string;
// }

// export interface WeatherRiskResponse {
//   lat:             number;
//   lng:             number;
//   weather_desc:    string;
//   temperature_c:   number | null;
//   visibility_m:    number | null;
//   wind_speed_mps:  number | null;
//   weather_risk:    number;
//   risk_label:      string;
//   advice:          string;
// }

// export interface DashboardStats {
//   total_blackspots:        number;
//   high_risk_count:         number;
//   medium_risk_count:       number;
//   low_risk_count:          number;
//   total_accidents_indexed: number;
//   total_killed_indexed:    number;
//   avg_risk_score:          number;
//   max_risk_score:          number;
//   min_risk_score:          number;
//   road_type_distribution:  Record<string, number>;
//   top_10_blackspots:       BlackSpot[];
// }

// export interface HealthResponse {
//   status:            string;
//   blackspots_loaded: number;
//   version:           string;
// }

// export interface Waypoint {
//   lat: number;
//   lng: number;
// }

// // ── API client ────────────────────────────────────────────────────────────────

// const BASE = CONFIG.API_URL;
// const TIMEOUT_MS = 8000;

// async function fetchWithTimeout(url: string, options?: RequestInit) {
//   const controller = new AbortController();
//   const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
//   try {
//     const res = await fetch(url, { ...options, signal: controller.signal });
//     if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
//     return await res.json();
//   } finally {
//     clearTimeout(timer);
//   }
// }

//  ── Endpoints ──────────

// /**
//  * Called every POLL_INTERVAL_MS as user moves.
//  * Returns alert status + nearby blackspots.
//  */
// export async function getNearby(
//   lat: number,
//   lng: number,
//   radius: number = CONFIG.NEARBY_RADIUS_M
// ): Promise<NearbyResponse> {
//   return fetchWithTimeout(
//     `${BASE}/api/v1/nearby?lat=${lat}&lng=${lng}&radius=${radius}`
//   );
// }

// /**
//  * Load all blackspot zones for the map heatmap.
//  * Called once on Map screen mount.
//  */
// export async function getAllBlackspots(
//   riskLevel?: "HIGH" | "MEDIUM" | "LOW"
// ): Promise<{ count: number; blackspots: BlackSpot[] }> {
//   const filter = riskLevel ? `&risk_level=${riskLevel}` : "";
//   return fetchWithTimeout(`${BASE}/api/v1/blackspots/all?limit=500${filter}`);
// }

// /**
//  * Dashboard statistics — accident counts, risk distribution, top zones.
//  */
// export async function getDashboardStats(): Promise<DashboardStats> {
//   return fetchWithTimeout(`${BASE}/api/v1/blackspots/stats`);
// }

// /**
//  * Single blackspot detail — when user taps a zone on the map.
//  */
// export async function getBlackspotById(
//   clusterId: number
// ): Promise<BlackSpot> {
//   return fetchWithTimeout(`${BASE}/api/v1/blackspots/${clusterId}`);
// }

// /**
//  * Score a route — list of waypoints → risk assessment.
//  */
// export async function scoreRoute(
//   waypoints: Waypoint[],
//   radiusM: number = 300
// ): Promise<RouteRiskResponse> {
//   return fetchWithTimeout(`${BASE}/api/v1/route-risk`, {
//     method:  "POST",
//     headers: { "Content-Type": "application/json" },
//     body:    JSON.stringify({ waypoints, radius_m: radiusM }),
//   });
// }

// /**
//  * Compare two routes — returns which is safer.
//  */
// export async function compareRoutes(
//   routeA: Waypoint[],
//   routeB: Waypoint[]
// ): Promise<RouteCompareResponse> {
//   return fetchWithTimeout(`${BASE}/api/v1/route-compare`, {
//     method:  "POST",
//     headers: { "Content-Type": "application/json" },
//     body:    JSON.stringify({ route_a: routeA, route_b: routeB }),
//   });
// }

// /**
//  * Live weather risk for current location.
//  */
// export async function getWeatherRisk(
//   lat: number,
//   lng: number
// ): Promise<WeatherRiskResponse> {
//   return fetchWithTimeout(
//     `${BASE}/api/v1/weather-risk?lat=${lat}&lng=${lng}`
//   );
// }

// /**
//  * Health check — used on app startup to confirm backend is reachable.
//  */
// export async function checkHealth(): Promise<HealthResponse> {
//   return fetchWithTimeout(`${BASE}/health`);
// }



/**
 * app/services/api.ts  ← FIXED VERSION
 * ───────────────────────────────────────
 * KEY FIXES:
 *   1. Removed AbortController — it causes "Aborted" errors in React Native
 *   2. Used Promise.race with a manual timeout promise instead
 *   3. Added detailed console.error logging so you can see exact failures
 *   4. Added API_URL validation on import
 *   5. All endpoints now log the exact URL being called
 */

import { CONFIG } from "../constants/config";

// ── Validate config on load ───────────────────────────────────────────────────
const BASE = CONFIG.API_URL.replace(/\/$/, ""); // strip trailing slash

if (__DEV__) {
  console.log(`[API] Base URL: ${BASE}`);
  if (BASE.includes("192.168.1.100")) {
    console.warn(
      "[API] ⚠️  Still using default IP 192.168.1.100 — " +
      "update API_URL in app/constants/config.ts to your laptop's actual IP"
    );
  }
}

// ── Types ─────────────────────────────────────────────────────────────────────
export interface BlackSpot {
  cluster_id:           number;
  lat:                  number;
  lng:                  number;
  accident_count:       number;
  total_killed:         number;
  avg_severity:         number;
  avg_env_risk:         number;
  state_weight:         number;
  night_accident_pct:   number;
  monsoon_accident_pct: number;
  primary_road_type:    string;
  top_weather:          string;
  risk_score:           number;
  risk_level:           "HIGH" | "MEDIUM" | "LOW";
  distance_m?:          number;
}

export interface NearbyResponse {
  alert:       boolean;
  alert_level: "HIGH" | "MEDIUM" | "SAFE";
  top_score:   number;
  black_spots: BlackSpot[];
  reason:      string;
  queried_at?: string;
}

export interface RouteRiskResponse {
  overall_risk_score:  number;
  risk_level:          string;
  total_blackspots:    number;
  high_risk_zones:     number;
  blackspots_on_route: BlackSpot[];
}

export interface RouteCompareResponse {
  route_a:         RouteRiskResponse;
  route_b:         RouteRiskResponse;
  safer_route:     "A" | "B";
  risk_difference: number;
  recommendation:  string;
}

export interface WeatherRiskResponse {
  lat:             number;
  lng:             number;
  weather_desc:    string;
  temperature_c:   number | null;
  visibility_m:    number | null;
  wind_speed_mps:  number | null;
  weather_risk:    number;
  risk_label:      string;
  advice:          string;
}

export interface DashboardStats {
  total_blackspots:        number;
  high_risk_count:         number;
  medium_risk_count:       number;
  low_risk_count:          number;
  total_accidents_indexed: number;
  total_killed_indexed:    number;
  avg_risk_score:          number;
  max_risk_score:          number;
  min_risk_score:          number;
  road_type_distribution:  Record<string, number>;
  top_10_blackspots:       BlackSpot[];
}

export interface HealthResponse {
  status:               string;
  blackspots_loaded:    number;
  blackspots_file?:     string;
  blackspots_file_exists?: boolean;
  version:              string;
  error?:               string | null;
}

export interface Waypoint {
  lat: number;
  lng: number;
}

// ── Core fetch helper ──────────────────────────────────────────────────────────
/**
 * FIXED: Uses Promise.race instead of AbortController.
 * AbortController causes "Aborted" errors in React Native's fetch polyfill.
 */
async function apiFetch(url: string, options?: RequestInit): Promise<any> {
  const TIMEOUT_MS = 15000; // 15 seconds — generous for local network

  if (__DEV__) {
    console.log(`[API] ${options?.method ?? "GET"} ${url}`);
  }

  const timeoutPromise = new Promise<never>((_, reject) =>
    setTimeout(() => reject(new Error(`Request timed out after ${TIMEOUT_MS / 1000}s — is the API server running?`)), TIMEOUT_MS)
  );

  const fetchPromise = fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      "Accept":       "application/json",
      ...(options?.headers ?? {}),
    },
  });

  try {
    const response = await Promise.race([fetchPromise, timeoutPromise]);

    if (!response.ok) {
      const body = await response.text().catch(() => "");
      throw new Error(`HTTP ${response.status} ${response.statusText}: ${body}`);
    }

    const data = await response.json();
    if (__DEV__) {
      console.log(`[API] ✅ ${url.split("?")[0]} →`, JSON.stringify(data).slice(0, 120));
    }
    return data;

  } catch (err: any) {
    console.error(`[API] ❌ ${url}:`, err.message);
    throw err;
  }
}

// ── Public API functions ───────────────────────────────────────────────────────

/** Health check — call this first to verify API is reachable */
export async function checkHealth(): Promise<HealthResponse> {
  return apiFetch(`${BASE}/health`);
}

/** Ping — fastest liveness check */
export async function ping(): Promise<boolean> {
  try {
    await apiFetch(`${BASE}/ping`);
    return true;
  } catch {
    return false;
  }
}

/** All blackspots for map heatmap */
export async function getAllBlackspots(
  riskLevel?: "HIGH" | "MEDIUM" | "LOW"
): Promise<{ count: number; blackspots: BlackSpot[] }> {
  const filter = riskLevel ? `&risk_level=${riskLevel}` : "";
  return apiFetch(`${BASE}/api/v1/blackspots/all?limit=500${filter}`);
}

/** Nearby blackspots + alert status — called on GPS position update */
export async function getNearby(
  lat:    number,
  lng:    number,
  radius: number = CONFIG.NEARBY_RADIUS_M
): Promise<NearbyResponse> {
  return apiFetch(
    `${BASE}/api/v1/nearby?lat=${lat}&lng=${lng}&radius=${radius}`
  );
}

/** Dashboard statistics */
export async function getDashboardStats(): Promise<DashboardStats> {
  return apiFetch(`${BASE}/api/v1/blackspots/stats`);
}

/** Single blackspot by ID */
export async function getBlackspotById(clusterId: number): Promise<BlackSpot> {
  return apiFetch(`${BASE}/api/v1/blackspots/${clusterId}`);
}

/** Score a single route */
export async function scoreRoute(
  waypoints: Waypoint[],
  radiusM: number = 300
): Promise<RouteRiskResponse> {
  return apiFetch(`${BASE}/api/v1/route-risk`, {
    method: "POST",
    body:   JSON.stringify({ waypoints, radius_m: radiusM }),
  });
}

/** Compare two routes */
export async function compareRoutes(
  routeA: Waypoint[],
  routeB: Waypoint[]
): Promise<RouteCompareResponse> {
  return apiFetch(`${BASE}/api/v1/route-compare`, {
    method: "POST",
    body:   JSON.stringify({ route_a: routeA, route_b: routeB }),
  });
}

/** Live weather risk */
export async function getWeatherRisk(
  lat: number,
  lng: number
): Promise<WeatherRiskResponse> {
  return apiFetch(`${BASE}/api/v1/weather-risk?lat=${lat}&lng=${lng}`);
}