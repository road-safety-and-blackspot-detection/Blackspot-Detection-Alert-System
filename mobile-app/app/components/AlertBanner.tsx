/**
 * app/components/AlertBanner.tsx
 * ────────────────────────────────
 * Inline banner shown at the top of screens when risk is MEDIUM.
 * For HIGH risk, the full AlertScreen takes over.
 */

import React from "react";
import { Animated, StyleSheet, Text, TouchableOpacity, View } from "react-native";
import { COLORS } from "../constants/colors";
import { AlertState } from "../hooks/useAlert";

interface AlertBannerProps {
  alertState: AlertState;
  onDismiss:  () => void;
  onTap?:     () => void;
}

export default function AlertBanner({
  alertState,
  onDismiss,
  onTap,
}: AlertBannerProps) {
  const isHigh   = alertState.level === "HIGH";
  if (!alertState.active || alertState.level === "HIGH") return null;

  const bgColor  = isHigh ? COLORS.highBg   : COLORS.mediumBg;
  const txtColor = isHigh ? COLORS.high     : COLORS.medium;
  const border   = isHigh ? COLORS.high     : COLORS.medium;

  return (
    <TouchableOpacity
      style={[styles.banner, { backgroundColor: bgColor, borderColor: border }]}
      onPress={onTap}
      activeOpacity={0.85}
    >
      <Text style={[styles.icon, { color: txtColor }]}>⚠</Text>
      <View style={styles.textBlock}>
        <Text style={[styles.title, { color: txtColor }]}>
          {alertState.level} RISK ZONE AHEAD
        </Text>
        <Text style={styles.reason} numberOfLines={1}>
          {alertState.reason}
        </Text>
      </View>
      <TouchableOpacity onPress={onDismiss} style={styles.closeBtn}>
        <Text style={[styles.closeText, { color: txtColor }]}>✕</Text>
      </TouchableOpacity>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  banner: {
    flexDirection:  "row",
    alignItems:     "center",
    marginHorizontal: 12,
    marginVertical:   6,
    padding:         12,
    borderRadius:    10,
    borderWidth:     1,
    gap:             10,
  },
  icon: {
    fontSize:   18,
    fontWeight: "700",
  },
  textBlock: {
    flex: 1,
    gap:  2,
  },
  title: {
    fontSize:   12,
    fontWeight: "800",
    letterSpacing: 0.5,
  },
  reason: {
    color:    COLORS.textSecondary,
    fontSize: 11,
  },
  closeBtn: {
    padding: 4,
  },
  closeText: {
    fontSize:   14,
    fontWeight: "700",
  },
});