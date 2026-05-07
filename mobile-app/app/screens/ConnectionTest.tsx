/**
 * app/screens/ConnectionTest.tsx
 * ────────────────────────────────
 * Debug screen — tests every API endpoint and shows exactly what's
 * working and what's failing. Run this first to diagnose issues.
 *
 * Access: Add a temporary button on any screen that calls:
 *   router.push("/screens/ConnectionTest")
 *
 * OR temporarily replace index.tsx with this screen.
 */

import React, { useState } from "react";
import {
  ActivityIndicator,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { COLORS } from "../constants/colors";
import { CONFIG } from "../constants/config";
import {
  checkHealth,
  getAllBlackspots,
  getDashboardStats,
  getNearby,
  getWeatherRisk,
  ping,
} from "../services/api";

interface TestResult {
  name:    string;
  status:  "idle" | "running" | "pass" | "fail";
  detail:  string;
  ms?:     number;
}

const INITIAL: TestResult[] = [
  { name: "Ping /ping",              status: "idle", detail: "" },
  { name: "Health /health",          status: "idle", detail: "" },
  { name: "All Blackspots /blackspots/all", status: "idle", detail: "" },
  { name: "Dashboard Stats /blackspots/stats", status: "idle", detail: "" },
  { name: "Nearby /nearby (Delhi)", status: "idle", detail: "" },
  { name: "Weather /weather-risk",   status: "idle", detail: "" },
];

const TESTS = [
  async () => {
    const ok = await ping();
    if (!ok) throw new Error("Ping returned false");
    return "pong ✓";
  },
  async () => {
    const h = await checkHealth();
    return `status=${h.status} | blackspots=${h.blackspots_loaded} | file_exists=${h.blackspots_file_exists}`;
  },
  async () => {
    const d = await getAllBlackspots();
    return `count=${d.count} | first_risk=${d.blackspots?.[0]?.risk_level ?? "n/a"}`;
  },
  async () => {
    const s = await getDashboardStats();
    return `total=${s.total_blackspots} high=${s.high_risk_count} med=${s.medium_risk_count}`;
  },
  async () => {
    // Delhi coords, 5km radius to maximise chance of finding zones
    const n = await getNearby(28.6139, 77.2090, 5000);
    return `alert=${n.alert} | level=${n.alert_level} | zones=${n.black_spots?.length ?? 0}`;
  },
  async () => {
    const w = await getWeatherRisk(28.6139, 77.2090);
    return `weather=${w.weather_desc} | risk=${w.weather_risk} | label=${w.risk_label}`;
  },
];

export default function ConnectionTest() {
  const [results, setResults] = useState<TestResult[]>(INITIAL);
  const [running, setRunning] = useState(false);

  const update = (i: number, patch: Partial<TestResult>) => {
    setResults((prev) => prev.map((r, idx) => idx === i ? { ...r, ...patch } : r));
  };

  const runAll = async () => {
    setRunning(true);
    setResults(INITIAL.map((r) => ({ ...r, status: "idle", detail: "" })));

    for (let i = 0; i < TESTS.length; i++) {
      update(i, { status: "running", detail: "Testing..." });
      const t0 = Date.now();
      try {
        const detail = await TESTS[i]();
        update(i, { status: "pass", detail, ms: Date.now() - t0 });
      } catch (err: any) {
        update(i, {
          status: "fail",
          detail: err.message ?? "Unknown error",
          ms:     Date.now() - t0,
        });
      }
    }
    setRunning(false);
  };

  const statusColor = (s: TestResult["status"]) => {
    if (s === "pass")    return COLORS.low;
    if (s === "fail")    return COLORS.high;
    if (s === "running") return COLORS.accent;
    return COLORS.textMuted;
  };

  const statusIcon = (s: TestResult["status"]) =>
    s === "pass" ? "✓" : s === "fail" ? "✗" : s === "running" ? "…" : "○";

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.title}>API Connection Test</Text>

      <View style={styles.urlBox}>
        <Text style={styles.urlLabel}>Testing against:</Text>
        <Text style={styles.url}>{CONFIG.API_URL}</Text>
        <Text style={styles.urlHint}>
          Change in app/constants/config.ts → API_URL
        </Text>
      </View>

      <TouchableOpacity
        style={[styles.runBtn, running && { opacity: 0.6 }]}
        onPress={runAll}
        disabled={running}
      >
        {running
          ? <ActivityIndicator color="#fff" />
          : <Text style={styles.runBtnText}>▶  Run All Tests</Text>
        }
      </TouchableOpacity>

      <View style={styles.resultsList}>
        {results.map((r, i) => (
          <View key={i} style={[
            styles.resultRow,
            r.status === "fail" && styles.resultRowFail,
            r.status === "pass" && styles.resultRowPass,
          ]}>
            <Text style={[styles.resultIcon, { color: statusColor(r.status) }]}>
              {statusIcon(r.status)}
            </Text>
            <View style={styles.resultInfo}>
              <Text style={styles.resultName}>{r.name}</Text>
              {r.detail ? (
                <Text style={[
                  styles.resultDetail,
                  r.status === "fail" && { color: COLORS.high },
                ]}>
                  {r.detail}
                </Text>
              ) : null}
              {r.ms !== undefined && (
                <Text style={styles.resultMs}>{r.ms}ms</Text>
              )}
            </View>
          </View>
        ))}
      </View>

      <View style={styles.helpBox}>
        <Text style={styles.helpTitle}>If all tests FAIL:</Text>
        <Text style={styles.helpText}>
          1. Make sure API is running:{"\n"}
          {"   "}cd BLACK-SPOT{"\n"}
          {"   "}uvicorn api.main:app --host 0.0.0.0 --port 8000{"\n"}
          {"\n"}
          2. Your phone + laptop must be on SAME WiFi{"\n"}
          {"\n"}
          3. Find laptop IP:{"\n"}
          {"   "}Windows: ipconfig → IPv4 Address{"\n"}
          {"   "}Mac: ipconfig getifaddr en0{"\n"}
          {"\n"}
          4. Set that IP in config.ts → API_URL{"\n"}
          {"\n"}
          5. Test in browser first:{"\n"}
          {"   "}http://YOUR_IP:8000/health
        </Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.background },
  content:   { padding: 16, gap: 14, paddingBottom: 40 },

  title: { color: COLORS.textPrimary, fontSize: 22, fontWeight: "800", marginTop: 8 },

  urlBox:    { backgroundColor: COLORS.surfaceRaised, borderRadius: 10,
               padding: 12, gap: 4, borderWidth: 1, borderColor: COLORS.border },
  urlLabel:  { color: COLORS.textSecondary, fontSize: 11 },
  url:       { color: COLORS.accent, fontSize: 15, fontWeight: "700",
               fontFamily: "monospace" },
  urlHint:   { color: COLORS.textMuted, fontSize: 10 },

  runBtn:     { backgroundColor: COLORS.accent, borderRadius: 10,
                paddingVertical: 14, alignItems: "center" },
  runBtnText: { color: "#fff", fontWeight: "800", fontSize: 16 },

  resultsList: { gap: 8 },
  resultRow:   { flexDirection: "row", backgroundColor: COLORS.surface,
                 borderRadius: 10, padding: 12, gap: 10, alignItems: "flex-start",
                 borderWidth: 1, borderColor: COLORS.border },
  resultRowPass: { borderColor: COLORS.low },
  resultRowFail: { borderColor: COLORS.high },
  resultIcon:    { fontSize: 18, fontWeight: "700", width: 20 },
  resultInfo:    { flex: 1, gap: 3 },
  resultName:    { color: COLORS.textPrimary, fontSize: 13, fontWeight: "700" },
  resultDetail:  { color: COLORS.textSecondary, fontSize: 11, fontFamily: "monospace" },
  resultMs:      { color: COLORS.textMuted, fontSize: 10 },

  helpBox:    { backgroundColor: COLORS.surfaceRaised, borderRadius: 10,
                padding: 14, borderWidth: 1, borderColor: COLORS.border },
  helpTitle:  { color: COLORS.textPrimary, fontWeight: "700", marginBottom: 6 },
  helpText:   { color: COLORS.textSecondary, fontSize: 12, lineHeight: 18,
                fontFamily: "monospace" },
});