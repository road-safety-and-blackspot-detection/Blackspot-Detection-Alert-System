/**
 * app/screens/DashboardScreen.tsx
 * ─────────────────────────────────
 * Analytics dashboard. Shows:
 *   • Risk level distribution (visual bar chart)
 *   • Total accidents + fatalities indexed
 *   • Road type breakdown
 *   • Top 10 most dangerous zones
 *   • Pull-to-refresh
 *
 * Extensible: add Chart.js, Victory Native, or react-native-gifted-charts
 * for richer visualisations. Currently uses pure RN for zero extra deps.
 */

import React, { useEffect, useState } from "react";
import {
  ActivityIndicator,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { useRouter } from "expo-router";

import { COLORS } from "../constants/colors";
import { getDashboardStats, DashboardStats } from "../services/api";
import RiskBadge from "../components/RiskBadge";

// ── Horizontal bar chart (pure RN — no deps) ─────────────────────────────────
function BarChart({
  data,
  total,
  color,
}: {
  data: { label: string; value: number }[];
  total: number;
  color: string;
}) {
  return (
    <View style={bar.container}>
      {data.map((item) => (
        <View key={item.label} style={bar.row}>
          <Text style={bar.label} numberOfLines={1}>{item.label}</Text>
          <View style={bar.track}>
            <View
              style={[
                bar.fill,
                {
                  width: `${Math.max((item.value / Math.max(total, 1)) * 100, 2)}%`,
                  backgroundColor: color,
                },
              ]}
            />
          </View>
          <Text style={bar.value}>{item.value}</Text>
        </View>
      ))}
    </View>
  );
}
const bar = StyleSheet.create({
  container: { gap: 8 },
  row:       { flexDirection: "row", alignItems: "center", gap: 8 },
  label:     { color: COLORS.textSecondary, fontSize: 11, width: 80 },
  track:     { flex: 1, height: 8, backgroundColor: COLORS.surfaceRaised,
               borderRadius: 4, overflow: "hidden" },
  fill:      { height: "100%", borderRadius: 4 },
  value:     { color: COLORS.textPrimary, fontSize: 11, fontWeight: "700", width: 36,
               textAlign: "right" },
});

// ── Stat card ─────────────────────────────────────────────────────────────────
function StatCard({
  label, value, sub, color,
}: { label: string; value: string | number; sub?: string; color?: string }) {
  return (
    <View style={sc.card}>
      <Text style={[sc.value, color ? { color } : {}]}>{value}</Text>
      <Text style={sc.label}>{label}</Text>
      {sub && <Text style={sc.sub}>{sub}</Text>}
    </View>
  );
}
const sc = StyleSheet.create({
  card:  { flex: 1, backgroundColor: COLORS.surfaceRaised, borderRadius: 10,
           padding: 14, alignItems: "center", gap: 4, minWidth: 90,
           borderWidth: 1, borderColor: COLORS.border },
  value: { color: COLORS.textPrimary, fontSize: 26, fontWeight: "900" },
  label: { color: COLORS.textSecondary, fontSize: 10, fontWeight: "600",
           textAlign: "center" },
  sub:   { color: COLORS.textMuted, fontSize: 9, textAlign: "center" },
});

// ── Main screen ───────────────────────────────────────────────────────────────
export default function DashboardScreen() {
  const router = useRouter();
  const [stats,      setStats]      = useState<DashboardStats | null>(null);
  const [isLoading,  setIsLoading]  = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error,      setError]      = useState<string | null>(null);

  const loadStats = async (silent = false) => {
    if (!silent) setIsLoading(true);
    setError(null);
    try {
      const data = await getDashboardStats();
      setStats(data);
    } catch (err: any) {
      setError(err.message ?? "Failed to load statistics");
    } finally {
      setIsLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => { loadStats(); }, []);

  const onRefresh = () => {
    setRefreshing(true);
    loadStats(true);
  };

  if (isLoading) return (
    <View style={styles.centred}>
      <ActivityIndicator color={COLORS.accent} size="large" />
      <Text style={styles.loadingText}>Loading statistics...</Text>
    </View>
  );

  if (error) return (
    <View style={styles.centred}>
      <Text style={styles.errorIcon}>⚡</Text>
      <Text style={styles.errorText}>{error}</Text>
      <TouchableOpacity style={styles.retryBtn} onPress={() => loadStats()}>
        <Text style={styles.retryText}>Retry</Text>
      </TouchableOpacity>
    </View>
  );

  if (!stats) return null;

  const total       = stats.total_blackspots;
  const riskData    = [
    { label: "HIGH",   value: stats.high_risk_count },
    { label: "MEDIUM", value: stats.medium_risk_count },
    { label: "LOW",    value: stats.low_risk_count },
  ];
  const roadData    = Object.entries(stats.road_type_distribution)
    .filter(([k]) => k !== "unknown")
    .sort(([, a], [, b]) => b - a)
    .slice(0, 6)
    .map(([label, value]) => ({ label, value }));
  const maxRoad     = Math.max(...roadData.map((d) => d.value), 1);

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
      showsVerticalScrollIndicator={false}
      refreshControl={
        <RefreshControl
          refreshing={refreshing}
          onRefresh={onRefresh}
          tintColor={COLORS.accent}
        />
      }
    >
      {/* ── Title ────────────────────────────────────────── */}
      <View style={styles.header}>
        <Text style={styles.title}>Analytics Dashboard</Text>
        <Text style={styles.subtitle}>
          Road safety intelligence powered by ML
        </Text>
      </View>

      {/* ── Top stats row ────────────────────────────────── */}
      <View style={styles.statsRow}>
        <StatCard label="Blackspots"   value={total}
                  sub="zones indexed" />
        <StatCard label="Accidents"    value={stats.total_accidents_indexed.toLocaleString()}
                  sub="total indexed" />
        <StatCard label="Fatalities"   value={stats.total_killed_indexed.toLocaleString()}
                  color={COLORS.high} sub="recorded" />
      </View>

      {/* ── Avg risk ─────────────────────────────────────── */}
      <View style={styles.avgCard}>
        <View style={styles.avgLeft}>
          <Text style={styles.avgLabel}>AVERAGE RISK SCORE</Text>
          <Text style={styles.avgValue}>{stats.avg_risk_score.toFixed(1)}</Text>
          <Text style={styles.avgRange}>
            Range: {stats.min_risk_score.toFixed(0)} – {stats.max_risk_score.toFixed(0)}
          </Text>
        </View>
        {/* Visual gauge */}
        <View style={styles.gaugeTrack}>
          <View style={[styles.gaugeFill, {
            width: `${stats.avg_risk_score}%`,
            backgroundColor: stats.avg_risk_score >= 70 ? COLORS.high
                           : stats.avg_risk_score >= 40 ? COLORS.medium
                           : COLORS.low,
          }]} />
        </View>
      </View>

      {/* ── Risk distribution ────────────────────────────── */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>RISK LEVEL DISTRIBUTION</Text>
        <View style={styles.riskBars}>
          {riskData.map(({ label, value }) => {
            const pct  = total > 0 ? value / total : 0;
            const color = label === "HIGH"   ? COLORS.high
                        : label === "MEDIUM" ? COLORS.medium
                        : COLORS.low;
            return (
              <View key={label} style={styles.riskBarItem}>
                <View style={[styles.riskBarFill, {
                  height: `${Math.max(pct * 100, 4)}%`,
                  backgroundColor: color,
                }]} />
                <Text style={[styles.riskBarLabel, { color }]}>{label}</Text>
                <Text style={styles.riskBarCount}>{value}</Text>
              </View>
            );
          })}
        </View>
      </View>

      {/* ── Road type breakdown ──────────────────────────── */}
      {roadData.length > 0 && (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>BLACKSPOTS BY ROAD TYPE</Text>
          <BarChart data={roadData} total={maxRoad} color={COLORS.accent} />
        </View>
      )}

      {/* ── Top 10 danger zones ──────────────────────────── */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>TOP 10 MOST DANGEROUS ZONES</Text>
        <View style={styles.topList}>
          {stats.top_10_blackspots.map((spot, idx) => (
            <View key={spot.cluster_id} style={styles.topItem}>
              <Text style={styles.topRank}>#{idx + 1}</Text>
              <View style={styles.topInfo}>
                <Text style={styles.topRoad}>
                  {spot.primary_road_type !== "unknown"
                    ? spot.primary_road_type.toUpperCase()
                    : `Zone ${spot.cluster_id}`}
                </Text>
                <Text style={styles.topCoords}>
                  {spot.lat.toFixed(3)}°N, {spot.lng.toFixed(3)}°E
                </Text>
                <Text style={styles.topMeta}>
                  {spot.accident_count} accidents · {spot.total_killed} fatalities
                </Text>
              </View>
              <RiskBadge level={spot.risk_level} score={spot.risk_score} size="sm" />
            </View>
          ))}
        </View>
      </View>

      {/* ── Footer note ──────────────────────────────────── */}
      <Text style={styles.footerNote}>
        Data indexed from India Mendeley crash records, Kaggle India
        accident dataset, and MoRTH official statistics.
        Pull to refresh.
      </Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.background },
  content:   { padding: 16, gap: 16, paddingBottom: 40 },
  centred:   { flex: 1, justifyContent: "center", alignItems: "center",
               backgroundColor: COLORS.background, gap: 12 },
  loadingText: { color: COLORS.textSecondary, fontSize: 13 },
  errorIcon:   { fontSize: 36 },
  errorText:   { color: COLORS.textSecondary, fontSize: 14, textAlign: "center" },
  retryBtn:    { backgroundColor: COLORS.accentBg, borderRadius: 8, paddingHorizontal: 20,
                 paddingVertical: 10, borderWidth: 1, borderColor: COLORS.accent },
  retryText:   { color: COLORS.accent, fontWeight: "700" },

  header:   { gap: 4 },
  title:    { color: COLORS.textPrimary, fontSize: 22, fontWeight: "800" },
  subtitle: { color: COLORS.textSecondary, fontSize: 13 },

  statsRow: { flexDirection: "row", gap: 8 },

  avgCard:    { backgroundColor: COLORS.surface, borderRadius: 14, padding: 16,
                borderWidth: 1, borderColor: COLORS.border, gap: 12 },
  avgLeft:    { gap: 2 },
  avgLabel:   { color: COLORS.textSecondary, fontSize: 10, fontWeight: "700",
                letterSpacing: 1.5 },
  avgValue:   { color: COLORS.textPrimary, fontSize: 40, fontWeight: "900" },
  avgRange:   { color: COLORS.textMuted, fontSize: 11 },
  gaugeTrack: { height: 8, backgroundColor: COLORS.surfaceRaised, borderRadius: 4,
                overflow: "hidden" },
  gaugeFill:  { height: "100%", borderRadius: 4 },

  card:      { backgroundColor: COLORS.surface, borderRadius: 14, padding: 16,
               borderWidth: 1, borderColor: COLORS.border, gap: 14 },
  cardTitle: { color: COLORS.textSecondary, fontSize: 10, fontWeight: "700",
               letterSpacing: 1.5 },

  riskBars:    { flexDirection: "row", height: 120, alignItems: "flex-end",
                 justifyContent: "space-around" },
  riskBarItem: { alignItems: "center", gap: 4, flex: 1 },
  riskBarFill: { width: 40, borderRadius: 4 },
  riskBarLabel:{ fontSize: 10, fontWeight: "700" },
  riskBarCount:{ color: COLORS.textSecondary, fontSize: 11 },

  topList:   { gap: 10 },
  topItem:   { flexDirection: "row", alignItems: "center", gap: 10,
               backgroundColor: COLORS.surfaceRaised, borderRadius: 10,
               padding: 12, borderWidth: 1, borderColor: COLORS.border },
  topRank:   { color: COLORS.textMuted, fontSize: 13, fontWeight: "700", width: 24 },
  topInfo:   { flex: 1, gap: 2 },
  topRoad:   { color: COLORS.textPrimary, fontSize: 13, fontWeight: "700" },
  topCoords: { color: COLORS.textSecondary, fontSize: 11 },
  topMeta:   { color: COLORS.textMuted, fontSize: 10 },

  footerNote: { color: COLORS.textMuted, fontSize: 11, textAlign: "center",
                lineHeight: 16 },
});