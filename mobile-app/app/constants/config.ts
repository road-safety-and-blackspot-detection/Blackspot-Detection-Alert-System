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

// export const CONFIG = {
//   // ── API connection ──────────────────────────────────────────
//   API_URL: "http://10.179.222.2:8000",   // ← CHANGE THIS to your laptop IP

//   // ── Alert thresholds ────────────────────────────────────────
//   HIGH_RISK_THRESHOLD:   70,   // triggers full-screen alert
//   MEDIUM_RISK_THRESHOLD: 40,   // triggers banner warning

//   // ── Location polling ────────────────────────────────────────
//   POLL_INTERVAL_MS:   5000,    // how often to check for nearby blackspots
//   NEARBY_RADIUS_M:    500,     // search radius in metres
//   ALERT_RADIUS_M:     300,     // tighter radius for triggering alerts

//   // ── Simulation ──────────────────────────────────────────────
//   SIMULATION_STEP_MS:    1500, // ms between each simulated position step
//   SIMULATION_ZOOM:       13,   // map zoom during simulation

//   // ── Map defaults ────────────────────────────────────────────
//   DEFAULT_LAT:   20.5937,      // India centre
//   DEFAULT_LNG:   78.9629,
//   DEFAULT_ZOOM:  5,

//   // ── Risk zone display radii ──────────────────────────────────
//   HIGH_ZONE_RADIUS_M:   250,
//   MEDIUM_ZONE_RADIUS_M: 200,
//   LOW_ZONE_RADIUS_M:    150,

//   // ── App version ──────────────────────────────────────────────
//   VERSION: "1.0.0",
// } as const;

// export type RiskLevel = "HIGH" | "MEDIUM" | "LOW" | "SAFE";



/**
 * app/constants/config.ts  ← FIXED VERSION
 * ──────────────────────────────────────────
 * HOW TO FIND YOUR LAPTOP IP:
 *
 *   Windows:
 *     1. Open CMD
 *     2. Type: ipconfig
 *     3. Look for "IPv4 Address" under your WiFi adapter
 *     4. It looks like: 192.168.x.x
 *
 *   Mac:
 *     1. Open Terminal
 *     2. Type: ipconfig getifaddr en0
 *     3. Or go to System Preferences → Network → WiFi → shows IP
 *
 *   Linux:
 *     1. Open Terminal
 *     2. Type: hostname -I | awk '{print $1}'
 *
 * IMPORTANT:
 *   - Your phone and laptop MUST be on the same WiFi network
 *   - Use HTTP not HTTPS
 *   - Do NOT use localhost or 127.0.0.1 (that's the phone itself, not laptop)
 *
 * EXAMPLE: If your laptop IP is 192.168.1.45, set:
 *   API_URL: "http://192.168.1.45:8000"
 */

export const CONFIG = {
  // ── ⚠️  CHANGE THIS to your laptop's WiFi IP address ──────────────
  API_URL: "http://192.168.1.7:8000",   // ← YOUR LAPTOP IP HERE

  // ── Alert thresholds ────────────────────────────────────────────────
  HIGH_RISK_THRESHOLD:   70,
  MEDIUM_RISK_THRESHOLD: 40,

  // ── Location polling ────────────────────────────────────────────────
  POLL_INTERVAL_MS:  5000,
  NEARBY_RADIUS_M:   500,
  ALERT_RADIUS_M:    300,

  // ── Simulation ──────────────────────────────────────────────────────
  SIMULATION_STEP_MS: 1500,
  SIMULATION_ZOOM:    13,

  // ── Map defaults (India centre) ─────────────────────────────────────
  DEFAULT_LAT:  20.5937,
  DEFAULT_LNG:  78.9629,
  DEFAULT_ZOOM: 5,

  // ── Risk zone display radii ──────────────────────────────────────────
  HIGH_ZONE_RADIUS_M:   250,
  MEDIUM_ZONE_RADIUS_M: 200,
  LOW_ZONE_RADIUS_M:    150,

  VERSION: "1.0.0",
} as const;

export type RiskLevel = "HIGH" | "MEDIUM" | "LOW" | "SAFE";