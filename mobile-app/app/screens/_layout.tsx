/**
 * app/screens/_layout.tsx
 * ────────────────────────
 * Stack navigator for screens pushed modally.
 */

import { Stack } from "expo-router";
import { COLORS } from "../constants/colors";

export default function ScreensLayout() {
  return (
    <Stack
      screenOptions={{
        headerStyle:      { backgroundColor: COLORS.surface },
        headerTintColor:   COLORS.textPrimary,
        headerTitleStyle:  { fontWeight: "700" },
        headerShadowVisible: false,
        contentStyle:      { backgroundColor: COLORS.background },
      }}
    >
      <Stack.Screen name="AlertScreen"     options={{ headerShown: false }} />
      <Stack.Screen name="RouteScreen"     options={{ title: "Route Planner" }} />
      <Stack.Screen name="DashboardScreen" options={{ title: "Analytics" }} />
      <Stack.Screen name="SimulationScreen" options={{ title: "Simulation Mode" }} />
    </Stack>
  );
}