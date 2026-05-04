/**
 * app/constants/colors.ts
 * ────────────────────────
 * Full design token system.
 * Dark-first theme — roads at night, danger red.
 * Every component imports from here — change once, updates everywhere.
 */

export const COLORS = {
  // ── Base palette ─────────────────────────────────────────────
  background:    "#0A0E1A",   // deep navy — map-like dark
  surface:       "#111827",   // slightly lighter card bg
  surfaceRaised: "#1C2333",   // elevated surface (modals, cards)
  border:        "#2A3448",   // subtle borders
  borderLight:   "#3A4560",   // lighter dividers

  // ── Text ─────────────────────────────────────────────────────
  textPrimary:   "#F0F4FF",   // near-white
  textSecondary: "#8B9AB8",   // muted blue-grey
  textMuted:     "#4A5568",   // very muted
  textOnDark:    "#FFFFFF",

  // ── Risk levels ───────────────────────────────────────────────
  high:          "#FF3B3B",   // danger red
  highBg:        "#2D0A0A",   // red tint background
  highGlow:      "rgba(255, 59, 59, 0.3)",
  medium:        "#FF8C00",   // warning orange
  mediumBg:      "#2D1A00",
  mediumGlow:    "rgba(255, 140, 0, 0.3)",
  low:           "#22C55E",   // safe green
  lowBg:         "#0A1F12",
  lowGlow:       "rgba(34, 197, 94, 0.3)",
  safe:          "#3B82F6",   // calm blue = no risk
  safeBg:        "#0A1428",

  // ── Map zone fill/stroke ──────────────────────────────────────
  highFill:      "rgba(255, 59, 59, 0.18)",
  highStroke:    "rgba(255, 59, 59, 0.8)",
  mediumFill:    "rgba(255, 140, 0, 0.15)",
  mediumStroke:  "rgba(255, 140, 0, 0.7)",
  lowFill:       "rgba(34, 197, 94, 0.12)",
  lowStroke:     "rgba(34, 197, 94, 0.6)",

  // ── Accent ───────────────────────────────────────────────────
  accent:        "#3B82F6",   // electric blue — interactive elements
  accentLight:   "#60A5FA",
  accentBg:      "rgba(59, 130, 246, 0.15)",

  // ── Status / UI ───────────────────────────────────────────────
  success:       "#22C55E",
  warning:       "#EAB308",
  error:         "#EF4444",
  info:          "#3B82F6",

  // ── Tab bar ──────────────────────────────────────────────────
  tabBarBg:      "#0D1220",
  tabActive:     "#3B82F6",
  tabInactive:   "#4A5568",

  // ── Gradients (used as array in LinearGradient) ───────────────
  gradientAlert:    ["#1A0A0A", "#2D0A0A"],
  gradientCard:     ["#111827", "#1C2333"],
  gradientOverlay:  ["transparent", "rgba(10,14,26,0.95)"],
} as const;

// Risk level → color mapping helpers
export const RISK_COLORS = {
  HIGH:   { text: COLORS.high,   bg: COLORS.highBg,   glow: COLORS.highGlow   },
  MEDIUM: { text: COLORS.medium, bg: COLORS.mediumBg, glow: COLORS.mediumGlow },
  LOW:    { text: COLORS.low,    bg: COLORS.lowBg,    glow: COLORS.lowGlow    },
  SAFE:   { text: COLORS.safe,   bg: COLORS.safeBg,   glow: COLORS.accentBg   },
} as const;

export const RISK_MAP_COLORS = {
  HIGH:   { fill: COLORS.highFill,   stroke: COLORS.highStroke   },
  MEDIUM: { fill: COLORS.mediumFill, stroke: COLORS.mediumStroke },
  LOW:    { fill: COLORS.lowFill,    stroke: COLORS.lowStroke    },
} as const;