/**
 * app/screens/RouteScreen.tsx
 * ────────────────────────────
 * Compares two routes (fast vs safe) by risk score.
 *
 * Features:
 *   • Enter origin + destination (text → geocode via OSM Nominatim, free)
 *   • Use current location as origin
 *   • Fetches two simulated route options from OSM OSRM (free routing API)
 *   • Scores both routes via /route-compare API
 *   • Shows winner, risk diff, and all danger zones along each route
 *   • Extensible: plug in Google Directions, HERE Maps, etc.
 */

import React, { useState } from "react";
import {
  ActivityIndicator,
  Alert,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import MapView, { Polyline, Circle, Marker } from "react-native-maps";

import { COLORS, RISK_MAP_COLORS } from "../constants/colors";
import { CONFIG } from "../constants/config";
import { compareRoutes, scoreRoute, RouteRiskResponse, Waypoint } from "../services/api";
import { useLocation } from "../hooks/useLocation";
import RiskBadge from "../components/RiskBadge";
import BlackspotCard from "../components/BlackspotCard";

// ── Free geocoding via OSM Nominatim ──────────────────────────────────────────
async function geocodePlace(query: string): Promise<{ lat: number; lng: number } | null> {
  try {
    const url = `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(
      query + ", India"
    )}&format=json&limit=1`;
    const res  = await fetch(url, {
      headers: { "User-Agent": "BlackSpotAlertApp/1.0" },
    });
    const data = await res.json();
    if (data.length === 0) return null;
    return { lat: parseFloat(data[0].lat), lng: parseFloat(data[0].lon) };
  } catch {
    return null;
  }
}

// ── Free routing via OSRM (OpenStreetMap) ────────────────────────────────────
async function getOSRMRoute(
  origin: Waypoint,
  dest:   Waypoint,
  profile: "driving" | "driving-short" = "driving"
): Promise<Waypoint[]> {
  try {
    const url =
      `https://router.project-osrm.org/route/v1/${profile}/` +
      `${origin.lng},${origin.lat};${dest.lng},${dest.lat}` +
      `?overview=full&geometries=geojson`;
    const res  = await fetch(url);
    const data = await res.json();
    const coords: [number, number][] = data.routes[0].geometry.coordinates;
    // OSRM returns [lng, lat] — flip to [lat, lng]
    return coords.map(([lng, lat]) => ({ lat, lng }));
  } catch {
    // Fallback: straight line with midpoint offset
    const midLat = (origin.lat + dest.lat) / 2 + (profile === "driving" ? 0.05 : -0.05);
    const midLng = (origin.lng + dest.lng) / 2;
    return [origin, { lat: midLat, lng: midLng }, dest];
  }
}

// ── Main component ────────────────────────────────────────────────────────────
export default function RouteScreen() {
  const { location } = useLocation();

  const [originText,  setOriginText]  = useState("");
  const [destText,    setDestText]    = useState("");
  const [isLoading,   setIsLoading]   = useState(false);
  const [error,       setError]       = useState<string | null>(null);

  const [routeA, setRouteA] = useState<Waypoint[]>([]);
  const [routeB, setRouteB] = useState<Waypoint[]>([]);
  const [resultA, setResultA] = useState<RouteRiskResponse | null>(null);
  const [resultB, setResultB] = useState<RouteRiskResponse | null>(null);
  const [saferRoute, setSaferRoute]     = useState<"A" | "B" | null>(null);
  const [recommendation, setRecommendation] = useState("");
  const [activeTab,  setActiveTab]    = useState<"A" | "B">("A");

  // ── Use current location as origin ───────────────────────────
  const useCurrentLocation = () => {
    if (!location) return;
    setOriginText(`${location.latitude.toFixed(4)}, ${location.longitude.toFixed(4)}`);
  };

  // ── Resolve origin coords ─────────────────────────────────────
  const resolveOrigin = async (): Promise<Waypoint | null> => {
    if (location && originText.trim() === "") return { lat: location.latitude, lng: location.longitude };
    // Check if it's raw coords
    const parts = originText.split(",");
    if (parts.length === 2) {
      const lat = parseFloat(parts[0]);
      const lng = parseFloat(parts[1]);
      if (!isNaN(lat) && !isNaN(lng)) return { lat, lng };
    }
    return geocodePlace(originText);
  };

  // ── Score routes ──────────────────────────────────────────────
  const handleCompare = async () => {
    if (!destText.trim()) {
      setError("Please enter a destination.");
      return;
    }
    setIsLoading(true);
    setError(null);
    setResultA(null); setResultB(null); setSaferRoute(null);

    try {
      const origin = await resolveOrigin();
      const dest   = await geocodePlace(destText);

      if (!origin) { setError("Could not find origin location."); return; }
      if (!dest)   { setError("Could not find destination. Try a more specific name."); return; }

      // Fetch two different OSRM routes (standard + shortest)
      const [wpsA, wpsB] = await Promise.all([
        getOSRMRoute(origin, dest, "driving"),
        getOSRMRoute(origin, dest, "driving-short"),
      ]);

      setRouteA(wpsA);
      setRouteB(wpsB);

      // Score both via backend
      const comparison = await compareRoutes(wpsA, wpsB);
      setResultA(comparison.route_a);
      setResultB(comparison.route_b);
      setSaferRoute(comparison.safer_route);
      setRecommendation(comparison.recommendation);
      setActiveTab(comparison.safer_route);
    } catch (err: any) {
      setError(err.message ?? "Failed to compare routes. Is the API running?");
    } finally {
      setIsLoading(false);
    }
  };

  const activeResult  = activeTab === "A" ? resultA : resultB;
  const activeWps     = activeTab === "A" ? routeA  : routeB;
  const activeColor   = activeTab === "A" ? COLORS.accent : COLORS.medium;

  const mapRegion = routeA.length > 0 ? {
    latitude:       (routeA[0].lat + routeA[routeA.length - 1].lat) / 2,
    longitude:      (routeA[0].lng + routeA[routeA.length - 1].lng) / 2,
    latitudeDelta:  Math.abs(routeA[0].lat - routeA[routeA.length - 1].lat) * 1.5 + 0.05,
    longitudeDelta: Math.abs(routeA[0].lng - routeA[routeA.length - 1].lng) * 1.5 + 0.05,
  } : undefined;

  return (
    <View style={styles.container}>
      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.scrollContent}
        keyboardShouldPersistTaps="handled"
        showsVerticalScrollIndicator={false}
      >
        {/* ── Header ──────────────────────────────────────── */}
        <View style={styles.header}>
          <Text style={styles.headerTitle}>Route Risk Planner</Text>
          <Text style={styles.headerSub}>
            Compare fast vs safe routes using ML blackspot data
          </Text>
        </View>

        {/* ── Input section ───────────────────────────────── */}
        <View style={styles.inputSection}>
          {/* Origin */}
          <View style={styles.inputRow}>
            <View style={styles.inputDot} />
            <TextInput
              style={styles.input}
              placeholder="Origin (or leave blank for current location)"
              placeholderTextColor={COLORS.textMuted}
              value={originText}
              onChangeText={setOriginText}
            />
            <TouchableOpacity style={styles.locBtn} onPress={useCurrentLocation}>
              <Text style={styles.locBtnText}>📍</Text>
            </TouchableOpacity>
          </View>

          {/* Route line */}
          <View style={styles.routeLine} />

          {/* Destination */}
          <View style={styles.inputRow}>
            <View style={[styles.inputDot, { backgroundColor: COLORS.high }]} />
            <TextInput
              style={styles.input}
              placeholder="Destination (e.g. Amritsar, Punjab)"
              placeholderTextColor={COLORS.textMuted}
              value={destText}
              onChangeText={setDestText}
            />
          </View>

          {error && <Text style={styles.errorText}>{error}</Text>}

          <TouchableOpacity
            style={[styles.compareBtn, isLoading && { opacity: 0.6 }]}
            onPress={handleCompare}
            disabled={isLoading}
          >
            {isLoading
              ? <ActivityIndicator color="#fff" />
              : <Text style={styles.compareBtnText}>⚡ Compare Routes</Text>
            }
          </TouchableOpacity>
        </View>

        {/* ── Map preview ─────────────────────────────────── */}
        {routeA.length > 0 && (
          <View style={styles.mapContainer}>
            <MapView style={styles.map} region={mapRegion ?? {
                latitude: 28.6139,
                longitude: 77.2090,
                latitudeDelta: 0.1,
                longitudeDelta: 0.1,
            }}>
              {/* Route A */}
              <Polyline
                coordinates={routeA.map((w) => ({ latitude: w.lat, longitude: w.lng }))}
                strokeColor={saferRoute === "A" ? COLORS.low : COLORS.accent}
                strokeWidth={saferRoute === "A" ? 5 : 3}
              />
              {/* Route B */}
              <Polyline
                coordinates={routeB.map((w) => ({ latitude: w.lat, longitude: w.lng }))}
                strokeColor={saferRoute === "B" ? COLORS.low : COLORS.medium}
                strokeWidth={saferRoute === "B" ? 5 : 3}
              />
              {/* Danger zones on active route */}
              {activeResult?.blackspots_on_route?.slice(0, 10).map((spot) => (
                <Circle
                  key={spot.cluster_id}
                  center={{ latitude: spot.lat, longitude: spot.lng }}
                  radius={200}
                //   fillColor={RISK_MAP_COLORS[spot.risk_level]?.fill}
                //   strokeColor={RISK_MAP_COLORS[spot.risk_level]?.stroke}
                     fillColor={RISK_MAP_COLORS[spot.risk_level]?.fill ?? "rgba(255,0,0,0.2)"}
                     strokeColor={RISK_MAP_COLORS[spot.risk_level]?.stroke ?? "#ff0000"}
                  strokeWidth={1.5}
                />
              ))}
            </MapView>
            {/* Route legend */}
            <View style={styles.mapLegend}>
              <Text style={[styles.legendItem, { color: COLORS.low }]}>
                ── Safer route
              </Text>
              <Text style={[styles.legendItem, { color: COLORS.accent }]}>
                ── Alternative
              </Text>
            </View>
          </View>
        )}

        {/* ── Results ─────────────────────────────────────── */}
        {(resultA || resultB) && (
          <View style={styles.resultsSection}>

            {/* Winner banner */}
            {saferRoute && (
              <View style={styles.winnerBanner}>
                <Text style={styles.winnerIcon}>🛡</Text>
                <View style={{ flex: 1 }}>
                  <Text style={styles.winnerTitle}>
                    Route {saferRoute} is SAFER
                  </Text>
                  <Text style={styles.winnerSub}>{recommendation}</Text>
                </View>
              </View>
            )}

            {/* Route tabs */}
            <View style={styles.tabBar}>
              {(["A", "B"] as const).map((tab) => {
                const res    = tab === "A" ? resultA : resultB;
                const isSafe = tab === saferRoute;
                return (
                  <TouchableOpacity
                    key={tab}
                    style={[styles.tab, activeTab === tab && styles.tabActive]}
                    onPress={() => setActiveTab(tab)}
                  >
                    <Text style={[styles.tabLabel, activeTab === tab && styles.tabLabelActive]}>
                      Route {tab} {isSafe ? "🛡" : ""}
                    </Text>
                    {res && (
                      <Text style={[styles.tabScore, {
                        color: res.risk_level === "HIGH"   ? COLORS.high
                             : res.risk_level === "MEDIUM" ? COLORS.medium
                             : COLORS.low,
                      }]}>
                        {res.overall_risk_score.toFixed(0)} risk
                      </Text>
                    )}
                  </TouchableOpacity>
                );
              })}
            </View>

            {/* Active route detail */}
            {activeResult && (
              <View style={styles.routeDetail}>
                {/* Summary cards */}
                <View style={styles.summaryRow}>
                  <View style={styles.summaryCard}>
                    <Text style={styles.summaryVal}>
                      {activeResult.overall_risk_score.toFixed(0)}
                    </Text>
                    <Text style={styles.summaryLbl}>Risk Score</Text>
                  </View>
                  <View style={styles.summaryCard}>
                    <Text style={[styles.summaryVal, { color: COLORS.high }]}>
                      {activeResult.high_risk_zones}
                    </Text>
                    <Text style={styles.summaryLbl}>HIGH Zones</Text>
                  </View>
                  <View style={styles.summaryCard}>
                    <Text style={styles.summaryVal}>
                      {activeResult.total_blackspots}
                    </Text>
                    <Text style={styles.summaryLbl}>Total Zones</Text>
                  </View>
                </View>

                <RiskBadge level={activeResult.risk_level as any} size="md" />

                {/* Danger zones list */}
                {activeResult.blackspots_on_route.length > 0 && (
                  <View style={styles.zonesSection}>
                    <Text style={styles.zonesSectionTitle}>
                      DANGER ZONES ON THIS ROUTE
                    </Text>
                    {activeResult.blackspots_on_route.slice(0, 5).map((spot) => (
                      <BlackspotCard key={spot.cluster_id} spot={spot} showDist={false} />
                    ))}
                    {activeResult.blackspots_on_route.length > 5 && (
                      <Text style={styles.moreZones}>
                        + {activeResult.blackspots_on_route.length - 5} more zones...
                      </Text>
                    )}
                  </View>
                )}

                {activeResult.blackspots_on_route.length === 0 && (
                  <View style={styles.cleanRoute}>
                    <Text style={styles.cleanIcon}>✓</Text>
                    <Text style={styles.cleanText}>No major blackspots on this route</Text>
                  </View>
                )}
              </View>
            )}
          </View>
        )}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container:    { flex: 1, backgroundColor: COLORS.background },
  scroll:       { flex: 1 },
  scrollContent: { padding: 16, gap: 16, paddingBottom: 40 },

  header: { gap: 4 },
  headerTitle: { color: COLORS.textPrimary, fontSize: 22, fontWeight: "800" },
  headerSub:   { color: COLORS.textSecondary, fontSize: 13 },

  inputSection: { backgroundColor: COLORS.surface, borderRadius: 14,
                  padding: 16, gap: 8, borderWidth: 1, borderColor: COLORS.border },
  inputRow:     { flexDirection: "row", alignItems: "center", gap: 10 },
  inputDot:     { width: 10, height: 10, borderRadius: 5,
                  backgroundColor: COLORS.accent },
  input:        { flex: 1, color: COLORS.textPrimary, fontSize: 14,
                  backgroundColor: COLORS.surfaceRaised, borderRadius: 8,
                  paddingHorizontal: 12, paddingVertical: 10,
                  borderWidth: 1, borderColor: COLORS.border },
  locBtn:       { padding: 8 },
  locBtnText:   { fontSize: 18 },
  routeLine:    { width: 2, height: 16, backgroundColor: COLORS.border, marginLeft: 4 },
  errorText:    { color: COLORS.error, fontSize: 12 },

  compareBtn:   { backgroundColor: COLORS.accent, borderRadius: 10,
                  paddingVertical: 14, alignItems: "center", marginTop: 4 },
  compareBtnText: { color: "#fff", fontWeight: "800", fontSize: 15 },

  mapContainer: { height: 220, borderRadius: 14, overflow: "hidden",
                  borderWidth: 1, borderColor: COLORS.border },
  map:          { flex: 1 },
  mapLegend:    { position: "absolute", bottom: 8, right: 8,
                  backgroundColor: "rgba(10,14,26,0.85)", borderRadius: 8,
                  padding: 8, gap: 2 },
  legendItem:   { fontSize: 11, fontWeight: "600" },

  resultsSection: { gap: 12 },

  winnerBanner: { flexDirection: "row", alignItems: "center",
                  backgroundColor: "rgba(34,197,94,0.1)",
                  borderWidth: 1, borderColor: COLORS.low,
                  borderRadius: 12, padding: 14, gap: 12 },
  winnerIcon:   { fontSize: 28 },
  winnerTitle:  { color: COLORS.low, fontWeight: "800", fontSize: 15 },
  winnerSub:    { color: COLORS.textSecondary, fontSize: 12, marginTop: 2 },

  tabBar:       { flexDirection: "row", gap: 8 },
  tab:          { flex: 1, backgroundColor: COLORS.surface, borderRadius: 10,
                  padding: 12, alignItems: "center",
                  borderWidth: 1, borderColor: COLORS.border },
  tabActive:    { borderColor: COLORS.accent, backgroundColor: COLORS.accentBg },
  tabLabel:     { color: COLORS.textSecondary, fontWeight: "700", fontSize: 14 },
  tabLabelActive: { color: COLORS.accent },
  tabScore:     { fontSize: 12, fontWeight: "600", marginTop: 2 },

  routeDetail:  { backgroundColor: COLORS.surface, borderRadius: 14,
                  padding: 16, gap: 14, borderWidth: 1, borderColor: COLORS.border },
  summaryRow:   { flexDirection: "row", gap: 8 },
  summaryCard:  { flex: 1, backgroundColor: COLORS.surfaceRaised, borderRadius: 10,
                  padding: 12, alignItems: "center", gap: 4 },
  summaryVal:   { color: COLORS.textPrimary, fontSize: 24, fontWeight: "900" },
  summaryLbl:   { color: COLORS.textSecondary, fontSize: 10, fontWeight: "600" },

  zonesSection:      { gap: 10 },
  zonesSectionTitle: { color: COLORS.textSecondary, fontSize: 10,
                       fontWeight: "700", letterSpacing: 1.5 },
  moreZones:         { color: COLORS.textMuted, fontSize: 12, textAlign: "center" },

  cleanRoute: { alignItems: "center", gap: 8, paddingVertical: 20 },
  cleanIcon:  { color: COLORS.low, fontSize: 36 },
  cleanText:  { color: COLORS.low, fontWeight: "700", fontSize: 15 },
});