import 'package:flutter/material.dart';

class RoiBoxPainter extends CustomPainter {
  final Rect? selectionRect;
  final List<List<dynamic>>? detectedRiskZones;

  RoiBoxPainter({this.selectionRect, this.detectedRiskZones});

  @override
  void paint(Canvas canvas, Size size) {
    // 1. Draw existing AI risk zones if available
    if (detectedRiskZones != null) {
      final zonePaint = Paint()
        ..color = const Color(0xFFEF4444).withOpacity(0.3)
        ..style = PaintingStyle.fill;
      final zoneBorder = Paint()
        ..color = const Color(0xFFEF4444)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.0;

      for (var z in detectedRiskZones!) {
        if (z.length == 4) {
          // Bbox [x0, y0, x1, y1] normalized to image width/height (usually 0-640, 0-480)
          final x0 = (z[0] as num).toDouble() / 640.0 * size.width;
          final y0 = (z[1] as num).toDouble() / 480.0 * size.height;
          final x1 = (z[2] as num).toDouble() / 640.0 * size.width;
          final y1 = (z[3] as num).toDouble() / 480.0 * size.height;
          final r = Rect.fromLTRB(x0, y0, x1, y1);
          canvas.drawRect(r, zonePaint);
          canvas.drawRect(r, zoneBorder);
        }
      }
    }

    // 2. Draw user active selection box
    if (selectionRect != null) {
      final fillPaint = Paint()
        ..color = const Color(0xFF38BDF8).withOpacity(0.2)
        ..style = PaintingStyle.fill;

      final borderPaint = Paint()
        ..color = const Color(0xFF38BDF8)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.5;

      canvas.drawRect(selectionRect!, fillPaint);
      canvas.drawRect(selectionRect!, borderPaint);

      // Draw corner grips
      final cornerPaint = Paint()..color = Colors.white;
      const cornerSize = 6.0;
      canvas.drawCircle(selectionRect!.topLeft, cornerSize, cornerPaint);
      canvas.drawCircle(selectionRect!.topRight, cornerSize, cornerPaint);
      canvas.drawCircle(selectionRect!.bottomLeft, cornerSize, cornerPaint);
      canvas.drawCircle(selectionRect!.bottomRight, cornerSize, cornerPaint);
    }
  }

  @override
  bool shouldRepaint(covariant RoiBoxPainter oldDelegate) {
    return oldDelegate.selectionRect != selectionRect ||
        oldDelegate.detectedRiskZones != detectedRiskZones;
  }
}
