/**
 * app/screens/MapScreen.tsx
 * ──────────────────────────
 * Primary screen. Shows:
 *   • User's live GPS location
 *   • All blackspot zones as coloured circles (HIGH=red, MEDIUM=orange, LOW=green)
 *   • Tap a zone → bottom sheet with full BlackspotCard detail
 *   • Live alert banner at top (MEDIUM risk)
 *   • Auto-navigates to AlertScreen for HIGH risk
 *
 * Extensible: add new map layers by adding <Circle> or <Marker> components.
 * Uses react-native-maps (free, uses device native maps).
 */

import React, { useCallback, useRef, useState } from "react";
import {
  ActivityIndicator,
  Animated,
  Dimensions,
  Modal,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import MapView, { Circle, Marker, Region } from "react-native-maps";
import { useRouter } from "expo-router";

import { COLORS, RISK_MAP_COLORS } from "../constants/colors";
import { CONFIG } from "../constants/config";
import { BlackSpot } from "../services/api";
import { useLocation } from "../hooks/useLocation";
import { useBlackspots } from "../hooks/useBlackspots";
import { useAlert } from "../hooks/useAlert";
import AlertBanner from "../components/AlertBanner";
import BlackspotCard from "../components/BlackspotCard";
import RiskBadge from "../components/RiskBadge";

const { height: SCREEN_H } = Dimensions.get("window");

export default function MapScreen() {
  const router = useRouter();
  const mapRef = useRef<MapView>(null);

  // ── Data hooks ──────────────────────────────────────────────
  const { location, hasPermission, isLoading: locLoading } = useLocation();
  const { blackspots, isLoading: spotsLoading, refetch }   = useBlackspots();
  const { alertState, dismissAlert }                        = useAlert(location);

  // ── Local state ─────────────────────────────────────────────
  const [selectedSpot, setSelectedSpot] = useState<BlackSpot | null>(null);
  const [filterLevel, setFilterLevel]   = useState<"ALL" | "HIGH" | "MEDIUM" | "LOW">("ALL");
  const sheetAnim = useRef(new Animated.Value(0)).current;

  // ── Navigate to alert screen when HIGH risk ──────────────────
  React.useEffect(() => {
    if (alertState.active && alertState.level === "HIGH") {
      router.push("/screens/AlertScreen");
    }
  }, [alertState.active, alertState.level]);

  // ── Centre map on user location when first acquired ──────────
  React.useEffect(() => {
    if (location && mapRef.current) {
      mapRef.current.animateToRegion(
        {
          latitude:       location.latitude,
          longitude:      location.longitude,
          latitudeDelta:  0.05,
          longitudeDelta: 0.05,
        },
        1000
      );
    }
  }, [location?.latitude, location?.longitude]);

  // ── Open bottom sheet for selected zone ─────────────────────
  const openSheet = useCallback((spot: BlackSpot) => {
    setSelectedSpot(spot);
    Animated.spring(sheetAnim, {
      toValue: 1, useNativeDriver: true,
    }).start();
  }, []);

  const closeSheet = useCallback(() => {
    Animated.timing(sheetAnim, {
      toValue: 0, duration: 250, useNativeDriver: true,
    }).start(() => setSelectedSpot(null));
  }, []);

  // ── Filter blackspots ────────────────────────────────────────
  const visibleSpots = filterLevel === "ALL"
    ? blackspots
    : blackspots.filter((s) => s.risk_level === filterLevel);

  // ── Zone radius by risk level ────────────────────────────────
  const zoneRadius = (level: string) => {
    if (level === "HIGH")   return CONFIG.HIGH_ZONE_RADIUS_M;
    if (level === "MEDIUM") return CONFIG.MEDIUM_ZONE_RADIUS_M;
    return CONFIG.LOW_ZONE_RADIUS_M;
  };

  if (!hasPermission && !locLoading) {
    return (
      <View style={styles.centred}>
        <Text style={styles.permText}>📍 Location permission required.</Text>
        <Text style={styles.permSub}>Enable location in your device Settings.</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>

      {/* ── Alert banner (MEDIUM risk) ─────────────────────── */}
      <AlertBanner
        alertState={alertState}
        onDismiss={dismissAlert}
        onTap={() => router.push("/screens/AlertScreen")}
      />

      {/* ── Filter bar ─────────────────────────────────────── */}
      <View style={styles.filterBar}>
        {(["ALL", "HIGH", "MEDIUM", "LOW"] as const).map((f) => (
          <TouchableOpacity
            key={f}
            style={[styles.filterBtn, filterLevel === f && styles.filterBtnActive]}
            onPress={() => setFilterLevel(f)}
          >
            <Text style={[
              styles.filterText,
              filterLevel === f && styles.filterTextActive,
            ]}>
              {f}
            </Text>
          </TouchableOpacity>
        ))}
        <TouchableOpacity style={styles.refreshBtn} onPress={refetch}>
          <Text style={styles.refreshText}>↻</Text>
        </TouchableOpacity>
      </View>

      {/* ── Map ────────────────────────────────────────────── */}
      <MapView
        ref={mapRef}
        style={styles.map}
        initialRegion={{
          latitude:       CONFIG.DEFAULT_LAT,
          longitude:      CONFIG.DEFAULT_LNG,
          latitudeDelta:  20,
          longitudeDelta: 20,
        }}
        // showsUserLocation={true}
        showsUserLocation
        // showsMyLocationButton={true}
        showsMyLocationButton
        mapType="standard"
        // customMapStyle={DARK_MAP_STYLE}
      >
        {/* Blackspot zones */}
        {visibleSpots.map((spot) => {
          const mc = RISK_MAP_COLORS[spot.risk_level] ?? RISK_MAP_COLORS.LOW;
          return (
            <>
            {/* <React.Fragment key={spot.cluster_id}> */}

              <Circle
                center={{ latitude: spot.lat, longitude: spot.lng }}
                radius={zoneRadius(spot.risk_level)}
                fillColor={mc.fill}
                strokeColor={mc.stroke}
                strokeWidth={1.5}
              />
              <Marker
                coordinate={{ latitude: spot.lat, longitude: spot.lng }}
                onPress={() => openSheet(spot)}
                anchor={{ x: 0.5, y: 0.5 }}
              >
                <View style={[
                  styles.zonePin,
                  { borderColor: mc.stroke }
                ]}>
                  <Text style={[styles.zonePinText, { color: mc.stroke }]}>
                    {spot.risk_level === "HIGH"   ? "⚠" :
                     spot.risk_level === "MEDIUM" ? "◆" : "●"}
                  </Text>
                </View>
              </Marker>
            {/* </React.Fragment> */}
            </>
          );
        })}
      </MapView>

      {/* ── Zone count badge ───────────────────────────────── */}
      <View style={styles.countBadge}>
        <Text style={styles.countText}>
          {spotsLoading ? "..." : `${visibleSpots.length} zones`}
        </Text>
      </View>

      {/* ── Loading overlay ────────────────────────────────── */}
      {(locLoading || spotsLoading) && (
        <View style={styles.loadingOverlay}>
          <ActivityIndicator color={COLORS.accent} size="large" />
          <Text style={styles.loadingText}>
            {locLoading ? "Getting location..." : "Loading blackspots..."}
          </Text>
        </View>
      )}

      {/* ── Bottom sheet: zone detail ───────────────────────── */}
      {selectedSpot && (
        <Modal
          visible={!!selectedSpot}
          transparent
          animationType="slide"
          onRequestClose={closeSheet}
        >
          <TouchableOpacity
            style={styles.sheetBackdrop}
            onPress={closeSheet}
            activeOpacity={1}
          />
          <View style={styles.sheet}>
            <View style={styles.sheetHandle} />
            <ScrollView showsVerticalScrollIndicator={false}>
              <BlackspotCard spot={selectedSpot} showDist={false} />
              <TouchableOpacity
                style={styles.routeBtn}
                onPress={() => {
                  closeSheet();
                  router.push("/screens/RouteScreen" as any);
                }}
              >
                <Text style={styles.routeBtnText}>📍 Plan Route Here</Text>
              </TouchableOpacity>
            </ScrollView>
          </View>
        </Modal>
      )}
    </View>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────
const styles = StyleSheet.create({
  container:    { flex: 1, backgroundColor: COLORS.background },
  map:          { flex: 1 },
  centred:      { flex: 1, justifyContent: "center", alignItems: "center",
                  backgroundColor: COLORS.background, padding: 32 },
  permText:     { color: COLORS.textPrimary, fontSize: 18, fontWeight: "700",
                  textAlign: "center" },
  permSub:      { color: COLORS.textSecondary, fontSize: 13, textAlign: "center",
                  marginTop: 8 },

  filterBar:    { flexDirection: "row", paddingHorizontal: 12, paddingVertical: 8,
                  backgroundColor: COLORS.surface, gap: 6,
                  borderBottomWidth: 1, borderColor: COLORS.border },
  filterBtn:    { paddingHorizontal: 10, paddingVertical: 5, borderRadius: 16,
                  backgroundColor: COLORS.surfaceRaised, borderWidth: 1,
                  borderColor: COLORS.border },
  filterBtnActive: { backgroundColor: COLORS.accentBg, borderColor: COLORS.accent },
  filterText:   { color: COLORS.textSecondary, fontSize: 11, fontWeight: "600" },
  filterTextActive: { color: COLORS.accent },
  refreshBtn:   { marginLeft: "auto", paddingHorizontal: 10, paddingVertical: 5 },
  refreshText:  { color: COLORS.accent, fontSize: 18 },

  zonePin:      { width: 28, height: 28, borderRadius: 14, borderWidth: 2,
                  backgroundColor: "rgba(10,14,26,0.75)",
                  justifyContent: "center", alignItems: "center" },
  zonePinText:  { fontSize: 12, fontWeight: "700" },

  countBadge:   { position: "absolute", bottom: 20, left: 16,
                  backgroundColor: COLORS.surfaceRaised, paddingHorizontal: 12,
                  paddingVertical: 6, borderRadius: 20, borderWidth: 1,
                  borderColor: COLORS.border },
  countText:    { color: COLORS.textSecondary, fontSize: 12 },

  loadingOverlay: { ...StyleSheet.absoluteFillObject, backgroundColor: "rgba(10,14,26,0.7)",
                    justifyContent: "center", alignItems: "center", gap: 12 },
  loadingText:    { color: COLORS.textSecondary, fontSize: 13 },

  sheetBackdrop: { flex: 1, backgroundColor: "rgba(0,0,0,0.5)" },
  sheet:        { backgroundColor: COLORS.surface, borderTopLeftRadius: 20,
                  borderTopRightRadius: 20, padding: 16, maxHeight: SCREEN_H * 0.55,
                  borderTopWidth: 1, borderColor: COLORS.border },
  sheetHandle:  { width: 40, height: 4, backgroundColor: COLORS.border,
                  borderRadius: 2, alignSelf: "center", marginBottom: 16 },
  routeBtn:     { backgroundColor: COLORS.accentBg, borderWidth: 1,
                  borderColor: COLORS.accent, borderRadius: 10, padding: 14,
                  alignItems: "center", marginTop: 12 },
  routeBtnText: { color: COLORS.accent, fontWeight: "700", fontSize: 14 },
});

// ── Dark map style (Google Maps JSON) ─────────────────────────────────────────
const DARK_MAP_STYLE = [
  { elementType: "geometry",        stylers: [{ color: "#0A0E1A" }] },
  { elementType: "labels.text.stroke", stylers: [{ color: "#0A0E1A" }] },
  { elementType: "labels.text.fill",   stylers: [{ color: "#8B9AB8" }] },
  { featureType: "road",              elementType: "geometry",       stylers: [{ color: "#1C2333" }] },
  { featureType: "road.highway",      elementType: "geometry",       stylers: [{ color: "#2A3448" }] },
  { featureType: "water",             elementType: "geometry",       stylers: [{ color: "#060A12" }] },
  { featureType: "poi",               elementType: "labels",         stylers: [{ visibility: "off" }] },
  { featureType: "transit",           elementType: "labels",         stylers: [{ visibility: "off" }] },
];