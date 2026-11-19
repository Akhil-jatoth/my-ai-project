import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:fl_chart/fl_chart.dart';
import '../models/prediction_model.dart';
import '../services/api_service.dart';
import '../theme/app_theme.dart';
import '../widgets/roi_box_painter.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _currentTabIndex = 0;
  final ImagePicker _picker = ImagePicker();

  // Scan state
  XFile? _selectedImage;
  PredictionResult? _prediction;
  bool _isAnalyzing = false;
  double _heatmapOpacity = 0.65;
  String? _errorMessage;

  // ROI Zone state
  bool _roiModeActive = false;
  Offset? _dragStart;
  Offset? _dragEnd;
  Rect? _activeRoiRect;
  String? _zoneCountText;
  bool _isCalculatingZone = false;

  // Video state
  XFile? _selectedVideo;
  VideoPredictionResult? _videoResult;
  bool _isVideoProcessing = false;
  String? _videoError;

  // System Health state
  SystemHealth? _health;

  // History state
  List<dynamic> _historyList = [];
  bool _isLoadingHistory = false;

  // Settings
  final TextEditingController _urlController = TextEditingController();

  @override
  void initState() {
    super.initState();
    _refreshHealth();
    _loadApiUrl();
  }

  Future<void> _loadApiUrl() async {
    final url = await ApiService.getBaseUrl();
    _urlController.text = url;
  }

  Future<void> _refreshHealth() async {
    final h = await ApiService.checkHealth();
    if (mounted) setState(() => _health = h);
  }

  // ── Image Processing ──────────────────────────────────────────────────────
  Future<void> _pickImage(ImageSource source) async {
    try {
      final XFile? file = await _picker.pickImage(
        source: source,
        maxWidth: 1920,
        maxHeight: 1080,
        imageQuality: 90,
      );
      if (file == null) return;

      setState(() {
        _selectedImage = file;
        _prediction = null;
        _errorMessage = null;
        _activeRoiRect = null;
        _zoneCountText = null;
        _isAnalyzing = true;
      });

      final result = await ApiService.predictImage(file);
      if (mounted) {
        setState(() {
          _prediction = result;
          _isAnalyzing = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _errorMessage = e.toString().replaceAll('Exception: ', '');
          _isAnalyzing = false;
        });
      }
    }
  }

  // ── Video Processing ──────────────────────────────────────────────────────
  Future<void> _pickVideo() async {
    try {
      final XFile? file = await _picker.pickVideo(
        source: ImageSource.gallery,
        maxDuration: const Duration(minutes: 3),
      );
      if (file == null) return;

      setState(() {
        _selectedVideo = file;
        _videoResult = null;
        _videoError = null;
        _isVideoProcessing = true;
      });

      final result = await ApiService.predictVideo(file);
      if (mounted) {
        setState(() {
          _videoResult = result;
          _isVideoProcessing = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _videoError = e.toString().replaceAll('Exception: ', '');
          _isVideoProcessing = false;
        });
      }
    }
  }

  // ── History ───────────────────────────────────────────────────────────────
  Future<void> _loadHistory() async {
    setState(() => _isLoadingHistory = true);
    final list = await ApiService.fetchHistory();
    if (mounted) {
      setState(() {
        _historyList = list;
        _isLoadingHistory = false;
      });
    }
  }

  Future<void> _clearHistory() async {
    final ok = await ApiService.clearHistory();
    if (ok) {
      setState(() => _historyList = []);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Scan history cleared successfully')),
        );
      }
    }
  }

  // ── ROI Zone Counting ─────────────────────────────────────────────────────
  Future<void> _onPanEnd(DragEndDetails details, Size displaySize) async {
    if (_prediction?.analysisId == null || _activeRoiRect == null) return;

    setState(() => _isCalculatingZone = true);
    try {
      // Map display coords to 640x480 normalized density grid
      final r0 = (_activeRoiRect!.top / displaySize.height * 480).toInt();
      final c0 = (_activeRoiRect!.left / displaySize.width * 640).toInt();
      final r1 = (_activeRoiRect!.bottom / displaySize.height * 480).toInt();
      final c1 = (_activeRoiRect!.right / displaySize.width * 640).toInt();

      final res = await ApiService.calculateZoneCount(
        _prediction!.analysisId!,
        [c0, r0, c1, r1],
      );

      if (mounted) {
        final count = res['zone_count_int'] ?? res['zone_count'] ?? 0;
        final pct = res['percentage_of_total'] ?? 0;
        setState(() {
          _zoneCountText = "Zone Count: $count ($pct% of total)";
          _isCalculatingZone = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _zoneCountText = "Error calculating zone";
          _isCalculatingZone = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Image.asset('assets/icons/app_icon.png', width: 28, height: 28),
            const SizedBox(width: 10),
            const Text("CrowdSense"),
          ],
        ),
        actions: [
          _buildHealthBadge(),
          IconButton(
            icon: const Icon(Icons.refresh, size: 20),
            onPressed: () {
              _refreshHealth();
              if (_currentTabIndex == 2) _loadHistory();
            },
          ),
        ],
      ),
      body: IndexedStack(
        index: _currentTabIndex,
        children: [
          _buildScanTab(),
          _buildVideoTab(),
          _buildHistoryTab(),
          _buildSettingsTab(),
        ],
      ),
      bottomNavigationBar: Container(
        decoration: BoxDecoration(
          border: Border(top: BorderSide(color: AppTheme.border, width: 1)),
        ),
        child: NavigationBar(
          selectedIndex: _currentTabIndex,
          backgroundColor: AppTheme.surface,
          indicatorColor: AppTheme.primary.withOpacity(0.25),
          onDestinationSelected: (i) {
            setState(() => _currentTabIndex = i);
            if (i == 2) _loadHistory();
          },
          destinations: const [
            NavigationDestination(
              icon: Icon(Icons.camera_alt_outlined),
              selectedIcon: Icon(Icons.camera_alt, color: AppTheme.accentCyan),
              label: 'Scan',
            ),
            NavigationDestination(
              icon: Icon(Icons.videocam_outlined),
              selectedIcon: Icon(Icons.videocam, color: AppTheme.accentCyan),
              label: 'Surveillance',
            ),
            NavigationDestination(
              icon: Icon(Icons.history_outlined),
              selectedIcon: Icon(Icons.history, color: AppTheme.accentCyan),
              label: 'History',
            ),
            NavigationDestination(
              icon: Icon(Icons.settings_outlined),
              selectedIcon: Icon(Icons.settings, color: AppTheme.accentCyan),
              label: 'Settings',
            ),
          ],
        ),
      ),
    );
  }

  // ── Health Status Badge ───────────────────────────────────────────────────
  Widget _buildHealthBadge() {
    final mode = _health?.mode.toUpperCase() ?? "CONNECTING";
    final isTrained = _health?.weightsLoaded ?? false;
    final color = isTrained ? AppTheme.lowRisk : AppTheme.moderateRisk;

    return Center(
      child: Container(
        margin: const EdgeInsets.only(right: 8),
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
        decoration: BoxDecoration(
          color: color.withOpacity(0.15),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: color.withOpacity(0.4)),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(width: 6, height: 6, decoration: BoxDecoration(color: color, shape: BoxShape.circle)),
            const SizedBox(width: 6),
            Text(
              mode,
              style: TextStyle(color: color, fontSize: 11, fontWeight: FontWeight.bold),
            ),
          ],
        ),
      ),
    );
  }

  // ── SCAN TAB ──────────────────────────────────────────────────────────────
  Widget _buildScanTab() {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        // Action Buttons Row
        Row(
          children: [
            Expanded(
              child: ElevatedButton.icon(
                icon: const Icon(Icons.camera_alt, size: 20),
                label: const Text("Camera"),
                onPressed: _isAnalyzing ? null : () => _pickImage(ImageSource.camera),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: ElevatedButton.icon(
                style: ElevatedButton.styleFrom(backgroundColor: AppTheme.surfaceLight),
                icon: const Icon(Icons.photo_library, size: 20),
                label: const Text("Gallery"),
                onPressed: _isAnalyzing ? null : () => _pickImage(ImageSource.gallery),
              ),
            ),
          ],
        ),
        const SizedBox(height: 16),

        // Error message if any
        if (_errorMessage != null)
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: AppTheme.criticalRisk.withOpacity(0.15),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: AppTheme.criticalRisk.withOpacity(0.5)),
            ),
            child: Row(
              children: [
                const Icon(Icons.error_outline, color: AppTheme.criticalRisk),
                const SizedBox(width: 10),
                Expanded(child: Text(_errorMessage!, style: const TextStyle(color: Colors.white, fontSize: 13))),
              ],
            ),
          ),

        // Image / Heatmap Display Card
        if (_selectedImage != null) _buildImagePreviewCard(),

        // Loading Indicator
        if (_isAnalyzing)
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 32),
            child: Column(
              children: [
                const CircularProgressIndicator(color: AppTheme.accentCyan),
                const SizedBox(height: 16),
                const Text("Deep Neural Density Estimation in progress...", style: TextStyle(color: AppTheme.textSecondary)),
              ],
            ),
          ),

        // Results Card
        if (_prediction != null && !_isAnalyzing) ...[
          const SizedBox(height: 16),
          _buildHeroCountCard(),
          const SizedBox(height: 16),
          _buildMetricsGrid(),
          if (_prediction!.riskZones.isNotEmpty) ...[
            const SizedBox(height: 16),
            _buildRiskZonesList(),
          ],
          if (_prediction!.recommendations.isNotEmpty) ...[
            const SizedBox(height: 16),
            _buildAdvisoriesCard(),
          ],
        ],

        if (_selectedImage == null && !_isAnalyzing)
          _buildScanPlaceholder(),
      ],
    );
  }

  Widget _buildScanPlaceholder() {
    return Container(
      margin: const EdgeInsets.only(top: 40),
      padding: const EdgeInsets.all(32),
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: AppTheme.border),
      ),
      child: Column(
        children: [
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: AppTheme.primary.withOpacity(0.1),
              shape: BoxShape.circle,
            ),
            child: const Icon(Icons.people_alt_outlined, size: 48, color: AppTheme.accentCyan),
          ),
          const SizedBox(height: 20),
          const Text(
            "Ready for Crowd Estimation",
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: AppTheme.textPrimary),
          ),
          const SizedBox(height: 8),
          const Text(
            "Capture a crowd photo with your camera or select an image from gallery to estimate real-time headcounts and risk zones.",
            textAlign: TextAlign.center,
            style: TextStyle(fontSize: 13, color: AppTheme.textSecondary, height: 1.4),
          ),
        ],
      ),
    );
  }

  Widget _buildImagePreviewCard() {
    return Card(
      clipBehavior: Clip.antiAlias,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          LayoutBuilder(
            builder: (context, constraints) {
              final w = constraints.maxWidth;
              final h = w * 0.75; // 4:3 aspect ratio

              return GestureDetector(
                onPanStart: _roiModeActive
                    ? (details) {
                        setState(() {
                          _dragStart = details.localPosition;
                          _dragEnd = details.localPosition;
                          _activeRoiRect = Rect.fromPoints(_dragStart!, _dragEnd!);
                        });
                      }
                    : null,
                onPanUpdate: _roiModeActive
                    ? (details) {
                        setState(() {
                          _dragEnd = details.localPosition;
                          _activeRoiRect = Rect.fromPoints(_dragStart!, _dragEnd!);
                        });
                      }
                    : null,
                onPanEnd: _roiModeActive ? (details) => _onPanEnd(details, Size(w, h)) : null,
                child: Stack(
                  children: [
                    // Base Image
                    Image.file(
                      File(_selectedImage!.path),
                      width: w,
                      height: h,
                      fit: BoxFit.cover,
                    ),

                    // Heatmap Overlay
                    if (_prediction?.heatmapBytes != null && _heatmapOpacity > 0.05)
                      Opacity(
                        opacity: _heatmapOpacity,
                        child: Image.memory(
                          _prediction!.heatmapBytes!,
                          width: w,
                          height: h,
                          fit: BoxFit.cover,
                        ),
                      ),

                    // Custom ROI Box Painter
                    CustomPaint(
                      size: Size(w, h),
                      painter: RoiBoxPainter(
                        selectionRect: _activeRoiRect,
                        detectedRiskZones: _prediction?.riskZones.map((z) => z.bbox).toList(),
                      ),
                    ),

                    // ROI Mode Indicator Tag
                    if (_roiModeActive)
                      Positioned(
                        top: 8,
                        left: 8,
                        child: Container(
                          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                          decoration: BoxDecoration(
                            color: Colors.black.withOpacity(0.7),
                            borderRadius: BorderRadius.circular(12),
                            border: Border.all(color: AppTheme.accentCyan),
                          ),
                          child: const Text("Drag rectangle to count zone", style: TextStyle(color: AppTheme.accentCyan, fontSize: 11)),
                        ),
                      ),
                  ],
                ),
              );
            },
          ),

          // Heatmap Controls & ROI Toggle
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            child: Row(
              children: [
                const Icon(Icons.layers_outlined, size: 18, color: AppTheme.textSecondary),
                const SizedBox(width: 8),
                const Text("Heatmap", style: TextStyle(fontSize: 12, color: AppTheme.textSecondary)),
                Expanded(
                  child: Slider(
                    value: _heatmapOpacity,
                    min: 0.0,
                    max: 1.0,
                    activeColor: AppTheme.accentCyan,
                    onChanged: (v) => setState(() => _heatmapOpacity = v),
                  ),
                ),
                TextButton.icon(
                  icon: Icon(_roiModeActive ? Icons.crop_free : Icons.crop, size: 16),
                  label: Text(_roiModeActive ? "Exit ROI" : "Zone ROI", style: const TextStyle(fontSize: 12)),
                  onPressed: () {
                    setState(() {
                      _roiModeActive = !_roiModeActive;
                      if (!_roiModeActive) {
                        _activeRoiRect = null;
                        _zoneCountText = null;
                      }
                    });
                  },
                ),
              ],
            ),
          ),

          // Zone calculation banner if active
          if (_zoneCountText != null || _isCalculatingZone)
            Container(
              color: AppTheme.surfaceLight,
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              child: Row(
                children: [
                  const Icon(Icons.location_searching, size: 16, color: AppTheme.accentCyan),
                  const SizedBox(width: 8),
                  Text(
                    _isCalculatingZone ? "Integrating density slice..." : _zoneCountText!,
                    style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 12, color: AppTheme.accentCyan),
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildHeroCountCard() {
    final riskColor = AppTheme.getRiskColor(_prediction!.level);

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [AppTheme.surface, AppTheme.surfaceLight],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: riskColor.withOpacity(0.4), width: 1.5),
        boxShadow: [
          BoxShadow(
            color: riskColor.withOpacity(0.12),
            blurRadius: 20,
            spreadRadius: 2,
          ),
        ],
      ),
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text("ESTIMATED HEADCOUNT", style: TextStyle(fontSize: 11, fontWeight: FontWeight.bold, letterSpacing: 1.2, color: AppTheme.textSecondary)),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                decoration: BoxDecoration(
                  color: riskColor.withOpacity(0.2),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: riskColor),
                ),
                child: Text("${_prediction!.level} Density", style: TextStyle(color: riskColor, fontWeight: FontWeight.bold, fontSize: 11)),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            "${_prediction!.count}",
            style: const TextStyle(
              fontSize: 54,
              fontWeight: FontWeight.w900,
              color: Colors.white,
              letterSpacing: -1.0,
            ),
          ),
          const SizedBox(height: 6),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.shield_outlined, size: 14, color: AppTheme.accentCyan),
              const SizedBox(width: 4),
              Text(
                "Uncertainty: ${_prediction!.uncertainty} (95% CI)",
                style: const TextStyle(color: AppTheme.textSecondary, fontSize: 12),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildMetricsGrid() {
    return Row(
      children: [
        Expanded(
          child: _buildMetricTile("Latency", "${_prediction!.processingTime.toStringAsFixed(0)} ms", Icons.speed),
        ),
        const SizedBox(width: 8),
        Expanded(
          child: _buildMetricTile("Model", _prediction!.model, Icons.memory),
        ),
        const SizedBox(width: 8),
        Expanded(
          child: _buildMetricTile("Risk Zones", "${_prediction!.riskZones.length}", Icons.warning_amber),
        ),
      ],
    );
  }

  Widget _buildMetricTile(String label, String value, IconData icon) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppTheme.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, size: 16, color: AppTheme.accentCyan),
          const SizedBox(height: 8),
          Text(value, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.white)),
          Text(label, style: const TextStyle(fontSize: 11, color: AppTheme.textMuted)),
        ],
      ),
    );
  }

  Widget _buildRiskZonesList() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text("Spatial Congestion Zones", style: TextStyle(fontWeight: FontWeight.bold, fontSize: 14)),
            const SizedBox(height: 12),
            ..._prediction!.riskZones.map((z) => Container(
                  margin: const EdgeInsets.only(bottom: 8),
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: AppTheme.surfaceLight,
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text("Zone ${z.zoneId} (${z.severity})", style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
                      Text("${z.areaPercentage.toStringAsFixed(1)}% of scene", style: const TextStyle(fontSize: 12, color: AppTheme.accentCyan)),
                    ],
                  ),
                )),
          ],
        ),
      ),
    );
  }

  Widget _buildAdvisoriesCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text("Safety Recommendations", style: TextStyle(fontWeight: FontWeight.bold, fontSize: 14)),
            const SizedBox(height: 8),
            ..._prediction!.recommendations.map((r) => Padding(
                  padding: const EdgeInsets.symmetric(vertical: 4),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text("• ", style: TextStyle(color: AppTheme.accentCyan, fontSize: 16)),
                      Expanded(child: Text(r, style: const TextStyle(fontSize: 12, color: AppTheme.textSecondary))),
                    ],
                  ),
                )),
          ],
        ),
      ),
    );
  }

  // ── VIDEO SURVEILLANCE TAB ────────────────────────────────────────────────
  Widget _buildVideoTab() {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        ElevatedButton.icon(
          icon: const Icon(Icons.video_file),
          label: const Text("Upload Surveillance Video (MP4 / AVI)"),
          onPressed: _isVideoProcessing ? null : _pickVideo,
        ),
        if (_selectedVideo != null) ...[
          const SizedBox(height: 8),
          Center(
            child: Text(
              "File: ${_selectedVideo!.name}",
              style: const TextStyle(fontSize: 12, color: AppTheme.accentCyan),
            ),
          ),
        ],
        const SizedBox(height: 16),

        if (_videoError != null)
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: AppTheme.criticalRisk.withOpacity(0.15),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Text(_videoError!, style: const TextStyle(color: Colors.white, fontSize: 13)),
          ),

        if (_isVideoProcessing)
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 32),
            child: Column(
              children: [
                const CircularProgressIndicator(color: AppTheme.accentCyan),
                const SizedBox(height: 16),
                const Text("Sampling frames and calculating density time series...", style: TextStyle(color: AppTheme.textSecondary)),
              ],
            ),
          ),

        if (_videoResult != null && !_isVideoProcessing) ...[
          // Stats Row
          Row(
            children: [
              Expanded(
                child: _buildMetricTile("Peak Crowd", "${_videoResult!.peakCount}", Icons.trending_up),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: _buildMetricTile("Avg Density", _videoResult!.avgCount.toStringAsFixed(1), Icons.show_chart),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: _buildMetricTile("Frames", "${_videoResult!.totalFramesProcessed}", Icons.filter_frames),
              ),
            ],
          ),
          const SizedBox(height: 20),

          // Temporal Timeline Chart
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text("Temporal Crowd Dynamics (Count vs Sec)", style: TextStyle(fontWeight: FontWeight.bold, fontSize: 14)),
                  const SizedBox(height: 24),
                  SizedBox(
                    height: 220,
                    child: LineChart(
                      LineChartData(
                        gridData: const FlGridData(show: true, drawVerticalLine: false),
                        titlesData: FlTitlesData(
                          topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                          rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                          bottomTitles: AxisTitles(
                            sideTitles: SideTitles(
                              showTitles: true,
                              getTitlesWidget: (val, _) => Text("${val.toInt()}s", style: const TextStyle(fontSize: 10, color: AppTheme.textMuted)),
                            ),
                          ),
                        ),
                        borderData: FlBorderData(show: false),
                        lineBarsData: [
                          LineChartBarData(
                            spots: _videoResult!.timeline
                                .map((p) => FlSpot(p.second, p.count.toDouble()))
                                .toList(),
                            isCurved: true,
                            color: AppTheme.accentCyan,
                            barWidth: 3,
                            dotData: const FlDotData(show: false),
                            belowBarData: BarAreaData(
                              show: true,
                              color: AppTheme.accentCyan.withOpacity(0.15),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],

        if (_videoResult == null && !_isVideoProcessing)
          Container(
            margin: const EdgeInsets.only(top: 40),
            padding: const EdgeInsets.all(32),
            decoration: BoxDecoration(
              color: AppTheme.surface,
              borderRadius: BorderRadius.circular(20),
            ),
            child: Column(
              children: [
                const Icon(Icons.videocam_outlined, size: 48, color: AppTheme.accentCyan),
                const SizedBox(height: 16),
                const Text("Surveillance Video Analytics", style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
                const SizedBox(height: 8),
                const Text(
                  "Upload a CCTV or security clip to track crowd influx and surge velocity over time.",
                  textAlign: TextAlign.center,
                  style: TextStyle(fontSize: 13, color: AppTheme.textSecondary),
                ),
              ],
            ),
          ),
      ],
    );
  }

  // ── HISTORY TAB ───────────────────────────────────────────────────────────
  Widget _buildHistoryTab() {
    if (_isLoadingHistory) {
      return const Center(child: CircularProgressIndicator());
    }

    if (_historyList.isEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.history_toggle_off, size: 54, color: AppTheme.textMuted),
            const SizedBox(height: 16),
            const Text("No past analyses recorded", style: TextStyle(color: AppTheme.textSecondary)),
            const SizedBox(height: 8),
            TextButton(onPressed: _loadHistory, child: const Text("Refresh")),
          ],
        ),
      );
    }

    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text("${_historyList.length} Total Records", style: const TextStyle(color: AppTheme.textSecondary, fontSize: 13)),
              TextButton.icon(
                icon: const Icon(Icons.delete_outline, size: 16, color: AppTheme.criticalRisk),
                label: const Text("Clear History", style: TextStyle(color: AppTheme.criticalRisk, fontSize: 12)),
                onPressed: _clearHistory,
              ),
            ],
          ),
        ),
        Expanded(
          child: ListView.builder(
            itemCount: _historyList.length,
            padding: const EdgeInsets.symmetric(horizontal: 16),
            itemBuilder: (context, index) {
              final item = _historyList[index];
              final count = item['crowd_count'] ?? item['count'] ?? 0;
              final level = item['density_level'] ?? 'Low';
              final time = item['timestamp'] ?? '';
              final color = AppTheme.getRiskColor(level);

              return Card(
                margin: const EdgeInsets.only(bottom: 8),
                child: ListTile(
                  leading: Container(
                    width: 44,
                    height: 44,
                    decoration: BoxDecoration(
                      color: color.withOpacity(0.15),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Center(
                      child: Text(
                        "$count",
                        style: TextStyle(fontWeight: FontWeight.bold, color: color),
                      ),
                    ),
                  ),
                  title: Text(item['filename'] ?? 'Camera Image', style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w600)),
                  subtitle: Text(time, style: const TextStyle(fontSize: 11, color: AppTheme.textMuted)),
                  trailing: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color: color.withOpacity(0.15),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Text(level, style: TextStyle(color: color, fontSize: 11, fontWeight: FontWeight.bold)),
                  ),
                ),
              );
            },
          ),
        ),
      ],
    );
  }

  // ── SETTINGS TAB ──────────────────────────────────────────────────────────
  Widget _buildSettingsTab() {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text("CrowdSense Backend Server URL", style: TextStyle(fontWeight: FontWeight.bold, fontSize: 14)),
                const SizedBox(height: 8),
                const Text(
                  "Enter the address of your Render deployment or local Flask server.",
                  style: TextStyle(fontSize: 12, color: AppTheme.textSecondary),
                ),
                const SizedBox(height: 16),
                TextField(
                  controller: _urlController,
                  decoration: InputDecoration(
                    hintText: "https://crowdsense.onrender.com",
                    filled: true,
                    fillColor: AppTheme.surfaceLight,
                    border: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: BorderSide.none),
                    prefixIcon: const Icon(Icons.link, color: AppTheme.accentCyan),
                  ),
                ),
                const SizedBox(height: 16),
                Row(
                  children: [
                    Expanded(
                      child: ElevatedButton(
                        child: const Text("Save URL"),
                        onPressed: () async {
                          await ApiService.setBaseUrl(_urlController.text);
                          await _refreshHealth();
                          if (mounted) {
                            ScaffoldMessenger.of(context).showSnackBar(
                              const SnackBar(content: Text('Backend URL saved successfully')),
                            );
                          }
                        },
                      ),
                    ),
                    const SizedBox(width: 8),
                    OutlinedButton(
                      child: const Text("Test Ping"),
                      onPressed: () async {
                        await _refreshHealth();
                        if (mounted) {
                          ScaffoldMessenger.of(context).showSnackBar(
                            SnackBar(
                              content: Text(_health?.isHealthy == true
                                  ? "Connection successful (${_health?.mode.toUpperCase()})"
                                  : "Could not connect to ${_urlController.text}"),
                            ),
                          );
                        }
                      },
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 16),

        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text("Diagnostics & Diagnostics Info", style: TextStyle(fontWeight: FontWeight.bold, fontSize: 14)),
                const SizedBox(height: 12),
                _buildDiagnosticRow("Model Architecture", _health?.model ?? 'MCNN'),
                _buildDiagnosticRow("Inference Hardware", _health?.device ?? 'CPU'),
                _buildDiagnosticRow("Weights Status", _health?.weightsLoaded == true ? "Loaded (weights.pth)" : "Demo Mode"),
                _buildDiagnosticRow("System Status", _health?.validationStatus ?? 'Unknown'),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildDiagnosticRow(String key, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(key, style: const TextStyle(fontSize: 12, color: AppTheme.textSecondary)),
          Text(value, style: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: Colors.white)),
        ],
      ),
    );
  }
}
