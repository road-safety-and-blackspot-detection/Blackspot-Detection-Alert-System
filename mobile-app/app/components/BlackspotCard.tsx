/**
 * app/components/BlackspotCard.tsx
 * ──────────────────────────────────
 * Displays detailed info for a single blackspot.
 * Used in bottom sheet on Map Screen and Route Screen.
 */

import React from "react";
import { StyleSheet, Text, View } from "react-native";
import { BlackSpot } from "../services/api";
import { COLORS } from "../constants/colors";
import RiskBadge from "./RiskBadge";

interface BlackspotCardProps {
  spot:     BlackSpot;
  showDist?: boolean;
}

function StatRow({ label, value }: { label: string; value: string | number }) {
  return (
    <View style={styles.statRow}>
      <Text style={styles.statLabel}>{label}</Text>
      <Text style={styles.statValue}>{value}</Text>
    </View>
  );
}

export default function BlackspotCard({ spot, showDist = false }: BlackspotCardProps) {
  return (
    <View style={styles.card}>
      {/* Header */}
      <View style={styles.header}>
        <View>
          <Text style={styles.title}>Accident Blackspot</Text>
          <Text style={styles.subtitle}>
            {spot.primary_road_type !== "unknown"
              ? spot.primary_road_type.toUpperCase()
              : "Road Zone"}
          </Text>
        </View>
        <RiskBadge level={spot.risk_level} score={spot.risk_score} size="lg" />
      </View>

      {/* Stats grid */}
      <View style={styles.statsGrid}>
        <StatRow label="🚗 Accidents"  value={spot.accident_count} />
        <StatRow label="💀 Fatalities" value={spot.total_killed} />
        <StatRow label="🌦 Weather"    value={spot.top_weather !== "unknown" ? spot.top_weather : "—"} />
        <StatRow label="🌙 Night risk" value={`${spot.night_accident_pct.toFixed(0)}%`} />
        <StatRow label="🌧 Monsoon"    value={`${spot.monsoon_accident_pct.toFixed(0)}%`} />
        {showDist && spot.distance_m !== undefined && (
          <StatRow label="📍 Distance" value={`${spot.distance_m}m away`} />
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: COLORS.surfaceRaised,
    borderRadius:    12,
    padding:         16,
    borderWidth:     1,
    borderColor:     COLORS.border,
    gap:             12,
  },
  header: {
    flexDirection:  "row",
    justifyContent: "space-between",
    alignItems:     "flex-start",
  },
  title: {
    color:      COLORS.textPrimary,
    fontSize:   16,
    fontWeight: "700",
  },
  subtitle: {
    color:    COLORS.textSecondary,
    fontSize: 11,
    marginTop: 2,
  },
  statsGrid: {
    gap: 6,
  },
  statRow: {
    flexDirection:  "row",
    justifyContent: "space-between",
    alignItems:     "center",
  },
  statLabel: {
    color:    COLORS.textSecondary,
    fontSize: 13,
  },
  statValue: {
    color:      COLORS.textPrimary,
    fontSize:   13,
    fontWeight: "600",
  },
});