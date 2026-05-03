/**
 * app/constants/config.ts
 * ─────────────────────────
 * Central configuration for the entire app.
 * Change API_URL to your laptop's local IP before running on device.
 *
 * Find your IP:
 *   Windows : ipconfig → IPv4 Address
 *   Mac/Linux: ifconfig → 192.168.x.x
 */

export const CONFIG = {
  // ── API connection ──────────────────────────────────────────
  API_URL: "http://192.168.1.4:8000",   // ← CHANGE THIS to your laptop IP

  // ── Alert thresholds ────────────────────────────────────────
  HIGH_RISK_THRESHOLD:   70,   // triggers full-screen alert
  MEDIUM_RISK_THRESHOLD: 40,   // triggers banner warning

  // ── Location polling ────────────────────────────────────────
  POLL_INTERVAL_MS:   5000,    // how often to check for nearby blackspots
  NEARBY_RADIUS_M:    500,     // search radius in metres
  ALERT_RADIUS_M:     300,     // tighter radius for triggering alerts

  // ── Simulation ──────────────────────────────────────────────
  SIMULATION_STEP_MS:    1500, // ms between each simulated position step
  SIMULATION_ZOOM:       13,   // map zoom during simulation

  // ── Map defaults ────────────────────────────────────────────
  DEFAULT_LAT:   20.5937,      // India centre
  DEFAULT_LNG:   78.9629,
  DEFAULT_ZOOM:  5,

  // ── Risk zone display radii ──────────────────────────────────
  HIGH_ZONE_RADIUS_M:   250,
  MEDIUM_ZONE_RADIUS_M: 200,
  LOW_ZONE_RADIUS_M:    150,

  // ── App version ──────────────────────────────────────────────
  VERSION: "1.0.0",
} as const;

export type RiskLevel = "HIGH" | "MEDIUM" | "LOW" | "SAFE";