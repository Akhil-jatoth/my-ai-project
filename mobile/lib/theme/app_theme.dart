import 'package:flutter/material.dart';

class AppTheme {
  static const Color background = Color(0xFF0A0E1A);
  static const Color surface = Color(0xFF131B2E);
  static const Color surfaceLight = Color(0xFF1E293B);
  static const Color cardBg = Color(0xFF162035);

  static const Color primary = Color(0xFF6366F1); // Indigo
  static const Color primaryGlow = Color(0xFF818CF8);
  static const Color accentCyan = Color(0xFF38BDF8); // Electric Cyan
  static const Color accentPurple = Color(0xFFA855F7); // Vivid Violet

  static const Color lowRisk = Color(0xFF10B981); // Emerald
  static const Color moderateRisk = Color(0xFFF59E0B); // Amber
  static const Color highRisk = Color(0xFFF97316); // Orange
  static const Color criticalRisk = Color(0xFFEF4444); // Crimson

  static const Color textPrimary = Color(0xFFF8FAFC);
  static const Color textSecondary = Color(0xFF94A3B8);
  static const Color textMuted = Color(0xFF64748B);
  static const Color border = Color(0xFF1E293B);

  static ThemeData get darkTheme {
    return ThemeData(
      brightness: Brightness.dark,
      scaffoldBackgroundColor: background,
      primaryColor: primary,
      colorScheme: const ColorScheme.dark(
        primary: primary,
        secondary: accentCyan,
        surface: surface,
        error: criticalRisk,
      ),
      appBarTheme: const AppBarTheme(
        backgroundColor: Colors.transparent,
        elevation: 0,
        centerTitle: true,
        titleTextStyle: TextStyle(
          color: textPrimary,
          fontSize: 18,
          fontWeight: FontWeight.bold,
          letterSpacing: 0.5,
        ),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: primary,
          foregroundColor: Colors.white,
          elevation: 4,
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
          textStyle: const TextStyle(fontWeight: FontWeight.w600, fontSize: 15),
        ),
      ),
    );
  }

  static Color getRiskColor(String? level) {
    switch (level?.toLowerCase()) {
      case 'low':
        return lowRisk;
      case 'moderate':
        return moderateRisk;
      case 'high':
        return highRisk;
      case 'critical':
        return criticalRisk;
      default:
        return accentCyan;
    }
  }
}
