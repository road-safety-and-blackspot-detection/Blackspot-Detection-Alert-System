/**
 * app/screens/AlertScreen.tsx
 * ────────────────────────────
 * Full-screen alert triggered when user enters a HIGH risk zone.
 *
 * Features:
 *   • Pulsing red background animation
 *   • Vibration pattern on mount
 *   • Shows risk score, reason, and zone details
 *   • "I Understand" dismiss button
 *   • Auto-dismisses if risk drops back to SAFE
 *   • Extensible: add sound via expo-av, add more details per zone
 */

import React, { useEffect, useRef } from "react";
import {
  Animated,
  Dimensions,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TouchableOpacity,
  Vibration,
  View,
} from "react-native";
import { useRouter } from "expo-router";
import * as Haptics from "expo-haptics";

import { COLORS } from "../constants/colors";
import { useLocation } from "../hooks/useLocation";
import { useAlert } from "../hooks/useAlert";
import BlackspotCard from "../components/BlackspotCard";

const { width: W, height: H } = Dimensions.get("window");

// Vibration pattern: long-short-short-long
const VIBRATION_PATTERN = [0, 400, 100, 400, 100, 800];

export default function AlertScreen() {
  const router = useRouter();
  const { location }               = useLocation();
  const { alertState, dismissAlert } = useAlert(location);

  // ── Animations ────────────────────────────────────────────────
  const pulseAnim  = useRef(new Animated.Value(0)).current;
  const scaleAnim  = useRef(new Animated.Value(0.85)).current;
  const opacityAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    // Entry haptics + vibration
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
    Vibration.vibrate(VIBRATION_PATTERN);

    // Fade + scale in
    Animated.parallel([
      Animated.spring(scaleAnim, { toValue: 1, useNativeDriver: true }),
      Animated.timing(opacityAnim, { toValue: 1, duration: 300, useNativeDriver: true }),
    ]).start();

    // Pulse background glow
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, { toValue: 1, duration: 900, useNativeDriver: false }),
        Animated.timing(pulseAnim, { toValue: 0, duration: 900, useNativeDriver: false }),
      ])
    ).start();

    return () => Vibration.cancel();
  }, []);

  // Auto-dismiss when area becomes safe
  useEffect(() => {
    if (!alertState.active && alertState.level === "SAFE") {
      handleDismiss();
    }
  }, [alertState.active, alertState.level]);

  const handleDismiss = () => {
    dismissAlert();
    Vibration.cancel();
    router.back();
  };

  const bgColor = pulseAnim.interpolate({
    inputRange:  [0, 1],
    outputRange: ["#0A0E1A", "#1A0505"],
  });

  const topSpots = alertState.response?.black_spots?.slice(0, 2) ?? [];

  return (
    <Animated.View style={[styles.container, { backgroundColor: bgColor }]}>
      <StatusBar barStyle="light-content" backgroundColor="#0A0E1A" />

      <ScrollView
        contentContainerStyle={styles.scroll}
        showsVerticalScrollIndicator={false}
      >
        <Animated.View
          style={[styles.content, {
            opacity: opacityAnim,
            transform: [{ scale: scaleAnim }],
          }]}
        >
          {/* ── Warning icon ──────────────────────────────── */}
          <View style={styles.iconWrapper}>
            <Animated.Text
              style={[styles.warningIcon, {
                transform: [{
                  scale: pulseAnim.interpolate({
                    inputRange: [0, 1], outputRange: [1, 1.12],
                  }),
                }],
              }]}
            >
              ⚠
            </Animated.Text>
          </View>

          {/* ── Title ────────────────────────────────────── */}
          <Text style={styles.title}>HIGH ACCIDENT ZONE</Text>
          <Text style={styles.subtitle}>AHEAD</Text>

          {/* ── Risk score pill ───────────────────────────── */}
          <View style={styles.scorePill}>
            <Text style={styles.scoreLabel}>RISK SCORE</Text>
            <Text style={styles.scoreValue}>
              {/* {alertState.topScore.toFixed(0)} */}
              {Number(alertState.topScore || 0).toFixed(0)}
              <Text style={styles.scoreMax}>/100</Text>
            </Text>
          </View>
``
          {/* ── Reason ───────────────────────────────────── */}
          {!!alertState.reason && (
            <View style={styles.reasonBox}>
              <Text style={styles.reasonTitle}>WHY THIS ZONE IS DANGEROUS</Text>
              <Text style={styles.reasonText}>{alertState.reason}</Text>
            </View>
          )}

          {/* ── Blackspot details ─────────────────────────── */}
          {topSpots.length > 0 && (
            <View style={styles.spotsSection}>
              <Text style={styles.sectionLabel}>ZONE DETAILS</Text>
              {topSpots.map((spot) => (
                <BlackspotCard key={spot.cluster_id} spot={spot} showDist />
              ))}
            </View>
          )}

          {/* ── Safety tips ───────────────────────────────── */}
          <View style={styles.tipsBox}>
            <Text style={styles.tipsTitle}>SAFETY TIPS</Text>
            {[
              "Slow down immediately",
              "Increase following distance",
              "Stay alert for pedestrians",
              "Avoid overtaking in this zone",
            ].map((tip, i) => (
              <Text key={i} style={styles.tip}>• {tip}</Text>
            ))}
          </View>

          {/* ── Dismiss button ────────────────────────────── */}
          <TouchableOpacity
            style={styles.dismissBtn}
            onPress={handleDismiss}
            activeOpacity={0.85}
          >
            <Text style={styles.dismissText}>✓  I UNDERSTAND — PROCEED WITH CAUTION</Text>
          </TouchableOpacity>

          <Text style={styles.footer}>
            Alert auto-clears when you leave the danger zone
          </Text>
        </Animated.View>
      </ScrollView>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  scroll:    { flexGrow: 1, paddingBottom: 40 },
  content:   { padding: 24, gap: 20, alignItems: "center" },

  iconWrapper: {
    width: 100, height: 100, borderRadius: 50,
    backgroundColor: "rgba(255,59,59,0.15)",
    borderWidth: 2, borderColor: COLORS.high,
    justifyContent: "center", alignItems: "center",
    marginTop: 32,
  },
  warningIcon: { fontSize: 48 },

  title:    { color: COLORS.high, fontSize: 32, fontWeight: "900",
              letterSpacing: 4, textAlign: "center" },
  subtitle: { color: COLORS.high, fontSize: 18, fontWeight: "300",
              letterSpacing: 8, marginTop: -12 },

  scorePill: {
    backgroundColor: COLORS.highBg, borderWidth: 1.5,
    borderColor: COLORS.high, borderRadius: 12,
    paddingHorizontal: 28, paddingVertical: 12, alignItems: "center",
  },
  scoreLabel: { color: COLORS.high, fontSize: 10, fontWeight: "700",
                letterSpacing: 2 },
  scoreValue: { color: COLORS.high, fontSize: 42, fontWeight: "900" },
  scoreMax:   { fontSize: 18, fontWeight: "400", color: COLORS.textSecondary },

  reasonBox: {
    backgroundColor: "rgba(255,59,59,0.08)", borderRadius: 10,
    padding: 16, width: "100%", borderLeftWidth: 3, borderLeftColor: COLORS.high,
  },
  reasonTitle: { color: COLORS.high, fontSize: 10, fontWeight: "700",
                 letterSpacing: 1.5, marginBottom: 6 },
  reasonText:  { color: COLORS.textPrimary, fontSize: 14, lineHeight: 20 },

  spotsSection: { width: "100%", gap: 10 },
  sectionLabel: { color: COLORS.textSecondary, fontSize: 10, fontWeight: "700",
                  letterSpacing: 1.5 },

  tipsBox: {
    backgroundColor: COLORS.surfaceRaised, borderRadius: 10,
    padding: 16, width: "100%", gap: 6,
  },
  tipsTitle: { color: COLORS.textSecondary, fontSize: 10, fontWeight: "700",
               letterSpacing: 1.5, marginBottom: 4 },
  tip: { color: COLORS.textPrimary, fontSize: 13, lineHeight: 20 },

  dismissBtn: {
    backgroundColor: COLORS.high, borderRadius: 12,
    paddingVertical: 16, paddingHorizontal: 24,
    width: "100%", alignItems: "center",
  },
  dismissText: { color: "#FFFFFF", fontSize: 13, fontWeight: "800",
                 letterSpacing: 0.5 },

  footer: { color: COLORS.textMuted, fontSize: 11, textAlign: "center" },
});