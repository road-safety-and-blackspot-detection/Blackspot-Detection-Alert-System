/**
 * app/services/api.ts
 * ────────────────────
 * Single source of truth for ALL API calls.
 * Every screen imports from here — never write fetch() directly in screens.
 *
 * If the API URL changes, or you add auth headers, you change it here only.
 */

import { CONFIG } from "../constants/config";

// ── Types (mirror backend Pydantic models) ────────────────────────────────────

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
  status:            string;
  blackspots_loaded: number;
  version:           string;
}

export interface Waypoint {
  lat: number;
  lng: number;
}

// ── API client ────────────────────────────────────────────────────────────────

const BASE = CONFIG.API_URL;
const TIMEOUT_MS = 8000;

async function fetchWithTimeout(url: string, options?: RequestInit) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

// ── Endpoints ─────────────────────────────────────────────────────────────────

/**
 * Called every POLL_INTERVAL_MS as user moves.
 * Returns alert status + nearby blackspots.
 */
export async function getNearby(
  lat: number,
  lng: number,
  radius: number = CONFIG.NEARBY_RADIUS_M
): Promise<NearbyResponse> {
  return fetchWithTimeout(
    `${BASE}/api/v1/nearby?lat=${lat}&lng=${lng}&radius=${radius}`
  );
}

/**
 * Load all blackspot zones for the map heatmap.
 * Called once on Map screen mount.
 */
export async function getAllBlackspots(
  riskLevel?: "HIGH" | "MEDIUM" | "LOW"
): Promise<{ count: number; blackspots: BlackSpot[] }> {
  const filter = riskLevel ? `&risk_level=${riskLevel}` : "";
  return fetchWithTimeout(`${BASE}/api/v1/blackspots/all?limit=500${filter}`);
}

/**
 * Dashboard statistics — accident counts, risk distribution, top zones.
 */
export async function getDashboardStats(): Promise<DashboardStats> {
  return fetchWithTimeout(`${BASE}/api/v1/blackspots/stats`);
}

/**
 * Single blackspot detail — when user taps a zone on the map.
 */
export async function getBlackspotById(
  clusterId: number
): Promise<BlackSpot> {
  return fetchWithTimeout(`${BASE}/api/v1/blackspots/${clusterId}`);
}

/**
 * Score a route — list of waypoints → risk assessment.
 */
export async function scoreRoute(
  waypoints: Waypoint[],
  radiusM: number = 300
): Promise<RouteRiskResponse> {
  return fetchWithTimeout(`${BASE}/api/v1/route-risk`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ waypoints, radius_m: radiusM }),
  });
}

/**
 * Compare two routes — returns which is safer.
 */
export async function compareRoutes(
  routeA: Waypoint[],
  routeB: Waypoint[]
): Promise<RouteCompareResponse> {
  return fetchWithTimeout(`${BASE}/api/v1/route-compare`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ route_a: routeA, route_b: routeB }),
  });
}

/**
 * Live weather risk for current location.
 */
export async function getWeatherRisk(
  lat: number,
  lng: number
): Promise<WeatherRiskResponse> {
  return fetchWithTimeout(
    `${BASE}/api/v1/weather-risk?lat=${lat}&lng=${lng}`
  );
}

/**
 * Health check — used on app startup to confirm backend is reachable.
 */
export async function checkHealth(): Promise<HealthResponse> {
  return fetchWithTimeout(`${BASE}/health`);
}