import 'dart:convert';
import 'dart:typed_data';

class PredictionResult {
  final int count;
  final String level;
  final double confidence;
  final String uncertainty;
  final String? heatmapBase64;
  final List<String> recommendations;
  final List<RiskZoneItem> riskZones;
  final String mode;
  final String model;
  final int? analysisId;
  final double processingTime;
  final Map<String, dynamic> metrics;

  PredictionResult({
    required this.count,
    required this.level,
    required this.confidence,
    required this.uncertainty,
    this.heatmapBase64,
    required this.recommendations,
    required this.riskZones,
    required this.mode,
    required this.model,
    this.analysisId,
    required this.processingTime,
    required this.metrics,
  });

  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    // Parse uncertainty
    String uStr = "±0";
    if (json['uncertainty'] != null) {
      uStr = json['uncertainty'].toString();
    } else if (json['prediction'] != null && json['prediction']['uncertainty_delta'] != null) {
      final delta = json['prediction']['uncertainty_delta'];
      uStr = "±$delta";
    }

    // Parse risk zones
    List<RiskZoneItem> zones = [];
    if (json['risk_zones'] != null && json['risk_zones']['zones'] != null) {
      for (var z in json['risk_zones']['zones']) {
        zones.add(RiskZoneItem.fromJson(z));
      }
    } else if (json['high_density_zones'] != null) {
      for (var z in json['high_density_zones']) {
        zones.add(RiskZoneItem.fromJson(z));
      }
    }

    // Parse recommendations
    List<String> recs = [];
    if (json['recommendations'] != null) {
      for (var r in json['recommendations']) {
        recs.add(r.toString());
      }
    }

    return PredictionResult(
      count: json['count'] is int ? json['count'] : (json['count'] as num?)?.round() ?? 0,
      level: json['level'] ?? 'Low',
      confidence: (json['confidence'] as num?)?.toDouble() ?? 85.0,
      uncertainty: uStr,
      heatmapBase64: json['heatmap_image'],
      recommendations: recs,
      riskZones: zones,
      mode: json['mode'] ?? 'trained',
      model: json['model'] ?? 'MCNN',
      analysisId: json['analysis_id'] is int ? json['analysis_id'] : int.tryParse(json['analysis_id']?.toString() ?? ''),
      processingTime: (json['processing_time'] as num?)?.toDouble() ?? 0.0,
      metrics: json['metrics'] is Map<String, dynamic> ? json['metrics'] : {},
    );
  }

  Uint8List? get heatmapBytes {
    if (heatmapBase64 == null) return null;
    try {
      final clean = heatmapBase64!.contains(',')
          ? heatmapBase64!.split(',').last
          : heatmapBase64!;
      return base64Decode(clean);
    } catch (_) {
      return null;
    }
  }
}

class RiskZoneItem {
  final int zoneId;
  final String severity;
  final double areaPercentage;
  final List<dynamic> bbox;

  RiskZoneItem({
    required this.zoneId,
    required this.severity,
    required this.areaPercentage,
    required this.bbox,
  });

  factory RiskZoneItem.fromJson(Map<String, dynamic> json) {
    return RiskZoneItem(
      zoneId: json['zone_id'] ?? json['zone'] ?? 1,
      severity: json['severity'] ?? 'Moderate',
      areaPercentage: (json['area_percentage'] as num?)?.toDouble() ?? 0.0,
      bbox: json['bbox'] is List ? json['bbox'] : [],
    );
  }
}

class VideoTimelinePoint {
  final double second;
  final int count;
  final String risk;

  VideoTimelinePoint({
    required this.second,
    required this.count,
    required this.risk,
  });

  factory VideoTimelinePoint.fromJson(Map<String, dynamic> json) {
    return VideoTimelinePoint(
      second: (json['timestamp_sec'] ?? json['second'] ?? json['frame_index'] ?? 0).toDouble(),
      count: (json['count'] as num?)?.round() ?? 0,
      risk: json['risk_level'] ?? 'Low',
    );
  }
}

class VideoPredictionResult {
  final bool success;
  final int peakCount;
  final double avgCount;
  final int totalFramesProcessed;
  final List<VideoTimelinePoint> timeline;

  VideoPredictionResult({
    required this.success,
    required this.peakCount,
    required this.avgCount,
    required this.totalFramesProcessed,
    required this.timeline,
  });

  factory VideoPredictionResult.fromJson(Map<String, dynamic> json) {
    List<VideoTimelinePoint> points = [];
    if (json['timeline'] != null) {
      for (var item in json['timeline']) {
        points.add(VideoTimelinePoint.fromJson(item));
      }
    }
    return VideoPredictionResult(
      success: json['success'] ?? false,
      peakCount: json['peak_count'] ?? 0,
      avgCount: (json['avg_count'] as num?)?.toDouble() ?? 0.0,
      totalFramesProcessed: json['frames_processed'] ?? points.length,
      timeline: points,
    );
  }
}

class SystemHealth {
  final bool isHealthy;
  final String mode;
  final String model;
  final String device;
  final bool weightsLoaded;
  final String validationStatus;

  SystemHealth({
    required this.isHealthy,
    required this.mode,
    required this.model,
    required this.device,
    required this.weightsLoaded,
    required this.validationStatus,
  });

  factory SystemHealth.fromJson(Map<String, dynamic> json) {
    return SystemHealth(
      isHealthy: json['status'] == 'healthy',
      mode: json['mode'] ?? 'demo',
      model: json['model'] ?? 'MCNN',
      device: json['device_label'] ?? json['device'] ?? 'CPU',
      weightsLoaded: json['weights_loaded'] ?? false,
      validationStatus: json['validation_status'] ?? 'Unknown',
    );
  }
}

