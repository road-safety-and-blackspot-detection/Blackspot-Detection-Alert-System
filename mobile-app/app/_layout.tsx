/**
 * app/_layout.tsx
 * ────────────────
 * Root layout — wraps entire app.
 * Sets up: status bar, safe area, navigation stack.
 */

import { Stack } from "expo-router";
import { StatusBar } from "expo-status-bar";
import { View } from "react-native";
import { SafeAreaProvider } from "react-native-safe-area-context";
import { COLORS } from "./constants/colors";

export default function RootLayout() {
  return (
    <SafeAreaProvider>
      <StatusBar style="light" backgroundColor={COLORS.background} />
      <Stack
        screenOptions={{
          headerStyle:    { backgroundColor: COLORS.surface },
          headerTintColor: COLORS.textPrimary,
          headerTitleStyle: { fontWeight: "700", fontSize: 16 },
          headerShadowVisible: false,
          contentStyle:   { backgroundColor: COLORS.background },
          animation:      "slide_from_right",
        }}
      >
        {/* Tab navigator — no header (tabs have their own) */}
        <Stack.Screen name="(tabs)" options={{ headerShown: false }} />

        {/* Alert screen — full screen modal */}
        <Stack.Screen
          name="screens/AlertScreen"
          options={{
            headerShown:  false,
            presentation: "fullScreenModal",
            animation:    "fade",
          }}
        />

        {/* Other screens pushed from tabs */}
        <Stack.Screen
          name="screens/RouteScreen"
          options={{ title: "Route Planner", headerShown: true }}
        />
        <Stack.Screen
          name="screens/DashboardScreen"
          options={{ title: "Analytics", headerShown: true }}
        />
        <Stack.Screen
          name="screens/SimulationScreen"
          options={{ title: "Simulation", headerShown: true }}
        />
      </Stack>
    </SafeAreaProvider>
  );
}