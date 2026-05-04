/**
 * app/screens/SimulationScreen.tsx
 * ──────────────────────────────────
 * SIMULATION MODE — The most important screen for demos.
 *
 * Why it matters:
 *   You don't need to physically drive to demo the app.
 *   Pick any two points, press Play, and the app simulates
 *   a vehicle moving along the route — triggering real alerts,
 *   updating the map, and showing heatmap changes in real time.
 *
 * Features:
 *   • Pick origin + destination by text (geocoded via OSM Nominatim)
 *   • Builds route via OSRM free routing API
 *   • Animates a moving marker along the route
 *   • Polls /nearby at each simulated position
 *   • Triggers AlertScreen (vibration + full-screen) for HIGH zones
 *   • Shows live risk indicator updating as vehicle moves
 *   • Play / Pause / Reset controls
 *   • Speed control (1x / 2x / 5x)
 *   • Extensible: add custom preset routes, record replay, etc.
 */

import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Animated,
  Dimensions,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  Vibration,
  View,
} from "react-native";
import MapView, { Circle, Marker, Polyline, Region } from "react-native-maps";
import * as Haptics from "expo-haptics";
import { useRouter } from "expo-router";

import { COLORS, RISK_MAP_COLORS } from "../constants/colors";
import { CONFIG } from "../constants/config";
import { getNearby, getAllBlackspots, BlackSpot, NearbyResponse } from "../services/api";
import RiskBadge from "../components/RiskBadge";
import BlackspotCard from "../components/BlackspotCard";

const { width: W, height: H } = Dimensions.get("window");

// ── Free geocoding ────────────────────────────────────────────────────────────
async function geocodePlace(query: string) {
  const url = `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(
    query + ", India"
  )}&format=json&limit=1`;
  const res  = await fetch(url, { headers: { "User-Agent": "BlackSpotAlertApp/1.0" } });
  const data = await res.json();
  if (!data.length) return null;
  return { lat: parseFloat(data[0].lat), lng: parseFloat(data[0].lon) };
}

// ── Free routing ──────────────────────────────────────────────────────────────
async function getRoute(
  origin: { lat: number; lng: number },
  dest:   { lat: number; lng: number }
) {
  const url =
    `https://router.project-osrm.org/route/v1/driving/` +
    `${origin.lng},${origin.lat};${dest.lng},${dest.lat}` +
    `?overview=full&geometries=geojson`;
  const res  = await fetch(url);
  const data = await res.json();
  const coords: [number, number][] = data.routes[0].geometry.coordinates;
  return coords.map(([lng, lat]) => ({ lat, lng }));
}

// ── Preset demo routes (India-specific) ───────────────────────────────────────
const PRESETS = [
  { label: "Ludhiana → Amritsar", from: "Ludhiana, Punjab", to: "Amritsar, Punjab" },
  { label: "Delhi → Jaipur",      from: "New Delhi",        to: "Jaipur, Rajasthan" },
  { label: "Mumbai → Pune",       from: "Mumbai",           to: "Pune, Maharashtra" },
  { label: "Bangalore → Mysore",  from: "Bangalore",        to: "Mysore, Karnataka" },
];

type SimState = "idle" | "loading" | "ready" | "playing" | "paused" | "done";

export default function SimulationScreen() {
  const router   = useRouter();
  const mapRef   = useRef<MapView>(null);

  // ── Input state ──────────────────────────────────────────────
  const [fromText,   setFromText]   = useState("");
  const [toText,     setToText]     = useState("");
  const [simState,   setSimState]   = useState<SimState>("idle");
  const [error,      setError]      = useState<string | null>(null);
  const [speed,      setSpeed]      = useState<1 | 2 | 5>(1);

  // ── Route data ───────────────────────────────────────────────
  const [routeWps,   setRouteWps]   = useState<{ lat: number; lng: number }[]>([]);
  const [blackspots, setBlackspots] = useState<BlackSpot[]>([]);

  // ── Simulation state ─────────────────────────────────────────
  const [stepIdx,    setStepIdx]    = useState(0);
  const [currentPos, setCurrentPos] = useState<{ lat: number; lng: number } | null>(null);
  const [nearbyResp, setNearbyResp] = useState<NearbyResponse | null>(null);
  const [alertShown, setAlertShown] = useState(false);
  const intervalRef  = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Log of triggered alerts during sim ───────────────────────
  const [alertLog, setAlertLog] = useState<
    { pos: number; level: string; score: number; reason: string }[]
  >([]);

  // ── Build route ───────────────────────────────────────────────
  const buildRoute = useCallback(async (from: string, to: string) => {
    setSimState("loading");
    setError(null);
    setAlertLog([]);
    try {
      const [origin, dest] = await Promise.all([
        geocodePlace(from),
        geocodePlace(to),
      ]);
      if (!origin) throw new Error(`Could not find: "${from}"`);
      if (!dest)   throw new Error(`Could not find: "${to}"`);

      const [route, spots] = await Promise.all([
        getRoute(origin, dest),
        getAllBlackspots(),
      ]);

      setRouteWps(route);
      setBlackspots(spots.blackspots ?? []);
      setStepIdx(0);
      setCurrentPos(route[0]);
      setSimState("ready");

      // Fit map to route
      if (mapRef.current && route.length > 1) {
        mapRef.current.fitToCoordinates(
          route.map((w) => ({ latitude: w.lat, longitude: w.lng })),
          { edgePadding: { top: 60, bottom: 60, left: 40, right: 40 }, animated: true }
        );
      }
    } catch (err: any) {
      setError(err.message ?? "Failed to build route");
      setSimState("idle");
    }
  }, []);

  // ── Step function: advance simulation one step ────────────────
  const tick = useCallback(async () => {
    setStepIdx((prev) => {
      const next = prev + 1;
      if (next >= routeWps.length) {
        // Done
        setSimState("done");
        if (intervalRef.current) clearInterval(intervalRef.current);
        return prev;
      }

      const pos = routeWps[next];
      setCurrentPos(pos);

      // Move map camera
      mapRef.current?.animateCamera(
        {
          center:  { latitude: pos.lat, longitude: pos.lng },
          zoom:    CONFIG.SIMULATION_ZOOM,
          heading: 0,
        },
        { duration: Math.max(300, CONFIG.SIMULATION_STEP_MS / speed - 200) }
      );

      // Poll API for alerts
      getNearby(pos.lat, pos.lng, CONFIG.ALERT_RADIUS_M)
        .then((resp) => {
          setNearbyResp(resp);
          if (resp.alert && resp.alert_level === "HIGH" && !alertShown) {
            setAlertShown(true);
            Vibration.vibrate([0, 400, 100, 400]);
            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
            setAlertLog((log) => [
              ...log,
              {
                pos:    next,
                level:  resp.alert_level,
                score:  resp.top_score,
                reason: resp.reason,
              },
            ]);
            // Show alert after brief delay so map update is visible
            setTimeout(() => {
              router.push("/screens/AlertScreen" as any);
              setAlertShown(false);
            }, 800);
          } else if (resp.alert && resp.alert_level === "MEDIUM") {
            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
            setAlertLog((log) => {
              // Avoid duplicate medium logs at same position
              if (log.length && log[log.length - 1].pos === next) return log;
              return [...log, { pos: next, level: resp.alert_level,
                                score: resp.top_score, reason: resp.reason }];
            });
          }
        })
        .catch(() => {}); // silent fail during sim

      return next;
    });
  }, [routeWps, speed, alertShown, router]);

  // ── Play / Pause / Reset ──────────────────────────────────────
  const play = () => {
    if (simState === "done") {
      setStepIdx(0);
      setCurrentPos(routeWps[0]);
      setAlertLog([]);
      setNearbyResp(null);
    }
    setSimState("playing");
    intervalRef.current = setInterval(tick, CONFIG.SIMULATION_STEP_MS / speed);
  };

  const pause = () => {
    setSimState("paused");
    if (intervalRef.current) clearInterval(intervalRef.current);
  };

  const reset = () => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    setStepIdx(0);
    setCurrentPos(routeWps.length > 0 ? routeWps[0] : null);
    setNearbyResp(null);
    setAlertLog([]);
    setSimState("ready");
  };

  // Restart interval when speed changes mid-play
  useEffect(() => {
    if (simState === "playing") {
      if (intervalRef.current) clearInterval(intervalRef.current);
      intervalRef.current = setInterval(tick, CONFIG.SIMULATION_STEP_MS / speed);
    }
  }, [speed, tick]);

  useEffect(() => {
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, []);

  // Progress 0–1
  const progress = routeWps.length > 1 ? stepIdx / (routeWps.length - 1) : 0;

  // Current alert color
  const alertLevel = nearbyResp?.alert_level ?? "SAFE";
  const alertColor = alertLevel === "HIGH"   ? COLORS.high
                   : alertLevel === "MEDIUM" ? COLORS.medium
                   : COLORS.low;

  return (
    <View style={styles.container}>

      {/* ── Map ──────────────────────────────────────────────── */}
      <MapView
        ref={mapRef}
        style={styles.map}
        initialRegion={{
          latitude: 20.5937, longitude: 78.9629,
          latitudeDelta: 20, longitudeDelta: 20,
        }}
        showsUserLocation={false}
        // scrollEnabled={simState !== "playing"}
        scrollEnabled={simState === "playing" ? false : true}
      >
        {/* Route polyline */}
        {routeWps.length > 1 && (
          <Polyline
            coordinates={routeWps.map((w) => ({ latitude: w.lat, longitude: w.lng }))}
            strokeColor={COLORS.accent}
            strokeWidth={3}
            lineDashPattern={[8, 4]}
          />
          
        )}

        {/* Travelled portion */}
        {stepIdx > 0 && routeWps.length > 1 && (
          <Polyline
            coordinates={routeWps.slice(0, stepIdx + 1).map((w) => ({
              latitude: w.lat, longitude: w.lng,
            }))}
            strokeColor={COLORS.low}
            strokeWidth={4}
          />
        )}

        {/* Blackspot zones */}
        {blackspots.map((spot) => (
          <Circle
            key={spot.cluster_id}
            center={{ latitude: spot.lat, longitude: spot.lng }}
            radius={CONFIG.MEDIUM_ZONE_RADIUS_M}
            // fillColor={RISK_MAP_COLORS[spot.risk_level]?.fill}
            // strokeColor={RISK_MAP_COLORS[spot.risk_level]?.stroke}
            fillColor={RISK_MAP_COLORS[spot.risk_level]?.fill ?? "rgba(255,0,0,0.2)"}
            strokeColor={RISK_MAP_COLORS[spot.risk_level]?.stroke ?? "#ff0000"}
            strokeWidth={1}
          />
        ))}

        {/* Simulated vehicle marker */}
        {currentPos && (
          <Marker
            coordinate={{ latitude: currentPos.lat, longitude: currentPos.lng }}
            anchor={{ x: 0.5, y: 0.5 }}
          >
            <View style={[
              styles.vehicleMarker,
              simState === "playing" && { borderColor: alertColor },
            ]}>
              <Text style={styles.vehicleIcon}>🚗</Text>
            </View>
          </Marker>
        )}

        {/* Destination marker */}
        {routeWps.length > 0 && (
          <Marker
            coordinate={{
              latitude:  routeWps[routeWps.length - 1].lat,
              longitude: routeWps[routeWps.length - 1].lng,
            }}
          >
            <View style={styles.destMarker}>
              <Text style={styles.destIcon}>🏁</Text>
            </View>
          </Marker>
        )}
      </MapView>

      {/* ── Live risk indicator (shown while playing) ────────── */}
      {simState === "playing" && nearbyResp && (
        <View style={[styles.liveRisk, { borderColor: alertColor }]}>
          <Text style={[styles.liveRiskLabel, { color: alertColor }]}>
            LIVE RISK
          </Text>
          <Text style={[styles.liveRiskScore, { color: alertColor }]}>
            {nearbyResp.top_score.toFixed(0)}
          </Text>
          <RiskBadge level={alertLevel as any} size="sm" />
        </View>
      )}

      {/* ── Bottom panel ─────────────────────────────────────── */}
      <ScrollView
        style={styles.panel}
        contentContainerStyle={styles.panelContent}
        keyboardShouldPersistTaps="handled"
        showsVerticalScrollIndicator={false}
      >

        {/* Setup section (shown when idle / ready) */}
        {(simState === "idle" || simState === "loading") && (
          <View style={styles.setupSection}>
            <Text style={styles.panelTitle}>Simulation Mode</Text>
            <Text style={styles.panelSub}>
              Simulate driving and watch alerts trigger in real time
            </Text>

            {/* Preset routes */}
            <Text style={styles.sectionLabel}>QUICK ROUTES</Text>
            <ScrollView horizontal showsHorizontalScrollIndicator={false}
              contentContainerStyle={styles.presetsRow}>
              {PRESETS.map((preset) => (
                <TouchableOpacity
                  key={preset.label}
                  style={styles.presetBtn}
                  onPress={() => {
                    setFromText(preset.from);
                    setToText(preset.to);
                    buildRoute(preset.from, preset.to);
                  }}
                >
                  <Text style={styles.presetText}>{preset.label}</Text>
                </TouchableOpacity>
              ))}
            </ScrollView>

            {/* Custom route inputs */}
            <Text style={styles.sectionLabel}>CUSTOM ROUTE</Text>
            <TextInput
              style={styles.input}
              placeholder="From (e.g. Ludhiana, Punjab)"
              placeholderTextColor={COLORS.textMuted}
              value={fromText}
              onChangeText={setFromText}
            />
            <TextInput
              style={styles.input}
              placeholder="To (e.g. Amritsar, Punjab)"
              placeholderTextColor={COLORS.textMuted}
              value={toText}
              onChangeText={setToText}
            />
            {error && <Text style={styles.errorText}>{error}</Text>}

            <TouchableOpacity
              style={[styles.buildBtn, simState === "loading" && { opacity: 0.6 }]}
              onPress={() => buildRoute(fromText, toText)}
              disabled={simState === "loading"}
            >
              {simState === "loading"
                ? <ActivityIndicator color="#fff" />
                : <Text style={styles.buildBtnText}>🗺  Build Route</Text>
              }
            </TouchableOpacity>
          </View>
        )}

        {/* Controls (shown when ready / playing / paused / done) */}
        {(simState === "ready" || simState === "playing" ||
          simState === "paused" || simState === "done") && (
          <View style={styles.controlSection}>
            {/* Progress bar */}
            <View style={styles.progressSection}>
              <View style={styles.progressTrack}>
                <View style={[styles.progressFill, { width: `${progress * 100}%` }]} />
              </View>
              <Text style={styles.progressText}>
                {stepIdx} / {routeWps.length - 1} steps
                {simState === "done" ? "  ✓ Done" : ""}
              </Text>
            </View>

            {/* Speed selector */}
            <View style={styles.speedRow}>
              <Text style={styles.speedLabel}>SPEED</Text>
              {([1, 2, 5] as const).map((s) => (
                <TouchableOpacity
                  key={s}
                  style={[styles.speedBtn, speed === s && styles.speedBtnActive]}
                  onPress={() => setSpeed(s)}
                >
                  <Text style={[styles.speedBtnText, speed === s && styles.speedBtnTextActive]}>
                    {s}×
                  </Text>
                </TouchableOpacity>
              ))}
            </View>

            {/* Play / Pause / Reset */}
            <View style={styles.controlBtns}>
              <TouchableOpacity style={styles.resetBtn} onPress={reset}>
                <Text style={styles.resetBtnText}>↺ Reset</Text>
              </TouchableOpacity>

              {simState === "playing" ? (
                <TouchableOpacity style={styles.playBtn} onPress={pause}>
                  <Text style={styles.playBtnText}>⏸  Pause</Text>
                </TouchableOpacity>
              ) : (
                <TouchableOpacity style={styles.playBtn} onPress={play}>
                  <Text style={styles.playBtnText}>
                    {simState === "done" ? "↺ Replay" : "▶  Play"}
                  </Text>
                </TouchableOpacity>
              )}

              <TouchableOpacity
                style={styles.rebuildBtn}
                onPress={() => setSimState("idle")}
              >
                <Text style={styles.rebuildBtnText}>✏ Edit</Text>
              </TouchableOpacity>
            </View>

            {/* Alert log */}
            {alertLog.length > 0 && (
              <View style={styles.alertLog}>
                <Text style={styles.alertLogTitle}>
                  ALERTS TRIGGERED ({alertLog.length})
                </Text>
                {alertLog.map((entry, i) => (
                  <View key={i} style={styles.alertLogItem}>
                    <Text style={[
                      styles.alertLogLevel,
                      { color: entry.level === "HIGH" ? COLORS.high : COLORS.medium },
                    ]}>
                      {entry.level === "HIGH" ? "⚠" : "◆"}  {entry.level}
                    </Text>
                    <Text style={styles.alertLogReason} numberOfLines={2}>
                      {entry.reason}
                    </Text>
                    <Text style={styles.alertLogStep}>at step {entry.pos}</Text>
                  </View>
                ))}
              </View>
            )}

            {/* Nearest blackspot during sim */}
            {simState === "playing" && nearbyResp?.black_spots?.[0] && (
              <View style={styles.nearestSection}>
                <Text style={styles.sectionLabel}>NEAREST ZONE</Text>
                <BlackspotCard
                  spot={nearbyResp.black_spots[0]}
                  showDist
                />
              </View>
            )}
          </View>
        )}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.background },
  map:       { height: H * 0.45 },

  liveRisk:  {
    position: "absolute", top: H * 0.45 - 70, right: 12,
    backgroundColor: COLORS.surface, borderRadius: 12, borderWidth: 2,
    padding: 10, alignItems: "center", gap: 4, minWidth: 70,
  },
  liveRiskLabel: { fontSize: 8, fontWeight: "800", letterSpacing: 1 },
  liveRiskScore: { fontSize: 26, fontWeight: "900" },

  panel:        { flex: 1, backgroundColor: COLORS.surface,
                  borderTopWidth: 1, borderColor: COLORS.border },
  panelContent: { padding: 16, gap: 14, paddingBottom: 40 },

  panelTitle: { color: COLORS.textPrimary, fontSize: 20, fontWeight: "800" },
  panelSub:   { color: COLORS.textSecondary, fontSize: 13 },

  setupSection:   { gap: 10 },
  sectionLabel:   { color: COLORS.textSecondary, fontSize: 10,
                    fontWeight: "700", letterSpacing: 1.5 },
  presetsRow:     { gap: 8, paddingBottom: 4 },
  presetBtn:      { backgroundColor: COLORS.surfaceRaised, borderRadius: 20,
                    paddingHorizontal: 14, paddingVertical: 8,
                    borderWidth: 1, borderColor: COLORS.border },
  presetText:     { color: COLORS.textPrimary, fontSize: 12, fontWeight: "600" },

  input:        { backgroundColor: COLORS.surfaceRaised, borderRadius: 10,
                  paddingHorizontal: 14, paddingVertical: 12, color: COLORS.textPrimary,
                  fontSize: 14, borderWidth: 1, borderColor: COLORS.border },
  errorText:    { color: COLORS.error, fontSize: 12 },
  buildBtn:     { backgroundColor: COLORS.accent, borderRadius: 10,
                  paddingVertical: 14, alignItems: "center" },
  buildBtnText: { color: "#fff", fontWeight: "800", fontSize: 15 },

  controlSection: { gap: 12 },
  progressSection:{ gap: 6 },
  progressTrack:  { height: 6, backgroundColor: COLORS.surfaceRaised,
                    borderRadius: 3, overflow: "hidden" },
  progressFill:   { height: "100%", backgroundColor: COLORS.accent, borderRadius: 3 },
  progressText:   { color: COLORS.textSecondary, fontSize: 11, textAlign: "right" },

  speedRow:         { flexDirection: "row", alignItems: "center", gap: 8 },
  speedLabel:       { color: COLORS.textSecondary, fontSize: 10, fontWeight: "700",
                      letterSpacing: 1, marginRight: 4 },
  speedBtn:         { paddingHorizontal: 12, paddingVertical: 6, borderRadius: 20,
                      backgroundColor: COLORS.surfaceRaised, borderWidth: 1,
                      borderColor: COLORS.border },
  speedBtnActive:   { backgroundColor: COLORS.accentBg, borderColor: COLORS.accent },
  speedBtnText:     { color: COLORS.textSecondary, fontWeight: "700", fontSize: 13 },
  speedBtnTextActive: { color: COLORS.accent },

  controlBtns:    { flexDirection: "row", gap: 8 },
  playBtn:        { flex: 1, backgroundColor: COLORS.accent, borderRadius: 10,
                    paddingVertical: 14, alignItems: "center" },
  playBtnText:    { color: "#fff", fontWeight: "800", fontSize: 15 },
  resetBtn:       { backgroundColor: COLORS.surfaceRaised, borderRadius: 10,
                    paddingVertical: 14, paddingHorizontal: 16,
                    borderWidth: 1, borderColor: COLORS.border },
  resetBtnText:   { color: COLORS.textSecondary, fontWeight: "700" },
  rebuildBtn:     { backgroundColor: COLORS.surfaceRaised, borderRadius: 10,
                    paddingVertical: 14, paddingHorizontal: 16,
                    borderWidth: 1, borderColor: COLORS.border },
  rebuildBtnText: { color: COLORS.textSecondary, fontWeight: "700" },

  alertLog:       { backgroundColor: COLORS.surfaceRaised, borderRadius: 12,
                    padding: 12, gap: 8, borderWidth: 1, borderColor: COLORS.border },
  alertLogTitle:  { color: COLORS.textSecondary, fontSize: 10, fontWeight: "700",
                    letterSpacing: 1.5 },
  alertLogItem:   { gap: 2, paddingVertical: 4, borderBottomWidth: 1,
                    borderColor: COLORS.border },
  alertLogLevel:  { fontWeight: "800", fontSize: 12 },
  alertLogReason: { color: COLORS.textSecondary, fontSize: 11 },
  alertLogStep:   { color: COLORS.textMuted, fontSize: 10 },

  nearestSection: { gap: 8 },

  vehicleMarker:  { width: 36, height: 36, borderRadius: 18, borderWidth: 2,
                    borderColor: COLORS.accent, backgroundColor: "rgba(10,14,26,0.85)",
                    justifyContent: "center", alignItems: "center" },
  vehicleIcon:    { fontSize: 18 },
  destMarker:     { width: 32, height: 32, justifyContent: "center",
                    alignItems: "center" },
  destIcon:       { fontSize: 24 },
});