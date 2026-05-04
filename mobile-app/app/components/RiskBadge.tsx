/**
 * app/components/RiskBadge.tsx
 * ─────────────────────────────
 * Compact coloured badge showing risk level.
 * Used in blackspot cards, list items, and the dashboard.
 */

import React from "react";
import { StyleSheet, Text, View } from "react-native";
import { COLORS, RISK_COLORS } from "../constants/colors";

interface RiskBadgeProps {
  level:  "HIGH" | "MEDIUM" | "LOW" | "SAFE";
  score?: number;       // optional numeric score to show alongside
  size?:  "sm" | "md" | "lg";
}

const LABELS = { HIGH: "⚠ HIGH", MEDIUM: "◆ MEDIUM", LOW: "● LOW", SAFE: "✓ SAFE" };
const SIZES  = {
  sm: { fontSize: 9,  paddingH: 6,  paddingV: 2, borderRadius: 3 },
  md: { fontSize: 11, paddingH: 8,  paddingV: 3, borderRadius: 4 },
  lg: { fontSize: 13, paddingH: 12, paddingV: 5, borderRadius: 6 },
};

export default function RiskBadge({
  level,
  score,
  size = "md",
}: RiskBadgeProps) {
  const color = RISK_COLORS[level] ?? RISK_COLORS.SAFE;
  const sz    = SIZES[size];

  return (
    <View
      style={[
        styles.badge,
        {
          backgroundColor:   color.bg,
          borderColor:        color.text,
          paddingHorizontal:  sz.paddingH,
          paddingVertical:    sz.paddingV,
          borderRadius:       sz.borderRadius,
        },
      ]}
    >
      <Text style={[styles.label, { color: color.text, fontSize: sz.fontSize }]}>
        {LABELS[level]}
        {score !== undefined ? `  ${score.toFixed(0)}` : ""}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  badge: {
    borderWidth:    1,
    alignSelf:      "flex-start",
    flexDirection:  "row",
    alignItems:     "center",
  },
  label: {
    fontWeight: "700",
    letterSpacing: 0.5,
  },
});