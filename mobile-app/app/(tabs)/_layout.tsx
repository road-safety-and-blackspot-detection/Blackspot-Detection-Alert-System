/**
 * app/(tabs)/_layout.tsx
 * ───────────────────────
 * Tab bar with 4 main tabs.
 * AlertScreen is NOT a tab — it's a modal pushed from the map.
 */

import { Tabs } from "expo-router";
import { COLORS } from "../constants/colors";
import { Text } from "react-native";
import { useSafeAreaInsets } from 'react-native-safe-area-context';
export default function TabLayout() {
   const insets = useSafeAreaInsets();
  return (
    <Tabs
      screenOptions={{
        tabBarStyle: {
          backgroundColor:  COLORS.tabBarBg,
          borderTopColor:   COLORS.border,
          borderTopWidth:   1,
          paddingTop:       6,
          paddingBottom: Math.max(insets.bottom, 10),
          height: 62 + Math.max(insets.bottom, 10),
        },
        tabBarActiveTintColor:   COLORS.tabActive,
        tabBarInactiveTintColor: COLORS.tabInactive,
        tabBarLabelStyle: {
          fontSize:   10,
          fontWeight: "700",
          letterSpacing: 0.5,
        },
        headerStyle:       { backgroundColor: COLORS.surface },
        headerTintColor:   COLORS.textPrimary,
        headerTitleStyle:  { fontWeight: "700" },
        headerShadowVisible: false,
      }}
    >
      <Tabs.Screen
        name="index"
        options={{
          title:       "Map",
          tabBarIcon:  ({ color }) => <TabIcon emoji="🗺" color={color} />,
          headerTitle: "Black Spot Alert",
        }}
      />
      <Tabs.Screen
        name="route"
        options={{
          title:       "Route",
          tabBarIcon:  ({ color }) => <TabIcon emoji="📍" color={color} />,
          headerTitle: "Route Planner",
        }}
      />
      <Tabs.Screen
        name="dashboard"
        options={{
          title:       "Analytics",
          tabBarIcon:  ({ color }) => <TabIcon emoji="📊" color={color} />,
          headerTitle: "Analytics Dashboard",
        }}
      />
      <Tabs.Screen
        name="simulation"
        options={{
          title:       "Simulate",
          tabBarIcon:  ({ color }) => <TabIcon emoji="▶" color={color} />,
          headerTitle: "Simulation Mode",
        }}
      />
    </Tabs>
  );
}

// Simple emoji tab icon — no icon library dependency needed
function TabIcon({ emoji, color }: { emoji: string; color: string }) {
//   const { Text } = require("react-native");
//   import { Text } from "react-native";
  return <Text style={{ fontSize: 20, opacity: color === COLORS.tabActive ? 1 : 0.5 }}>{emoji}</Text>;
}