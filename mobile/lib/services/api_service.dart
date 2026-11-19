import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/prediction_model.dart';

class ApiService {
  static const String _defaultUrl = "http://10.0.2.2:5000"; // Default Android emulator local URL
  static const String _prefUrlKey = "crowdsense_api_url";

  static Future<String> getBaseUrl() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_prefUrlKey) ?? _defaultUrl;
  }

  static Future<void> setBaseUrl(String url) async {
    final prefs = await SharedPreferences.getInstance();
    String cleanUrl = url.trim();
    if (cleanUrl.endsWith('/')) {
      cleanUrl = cleanUrl.substring(0, cleanUrl.length - 1);
    }
    await prefs.setString(_prefUrlKey, cleanUrl);
  }

  // 1. Health & Mode Check
  static Future<SystemHealth> checkHealth() async {
    final baseUrl = await getBaseUrl();
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/api/health'))
          .timeout(const Duration(seconds: 8));
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return SystemHealth.fromJson(data);
      }
      return SystemHealth(
        isHealthy: false,
        mode: 'offline',
        model: 'None',
        device: 'None',
        weightsLoaded: false,
        validationStatus: 'Server returned ${response.statusCode}',
      );
    } catch (e) {
      return SystemHealth(
        isHealthy: false,
        mode: 'unreachable',
        model: 'None',
        device: 'None',
        weightsLoaded: false,
        validationStatus: 'Cannot connect to backend: $e',
      );
    }
  }

  // 2. Predict Image Crowd Density
  static Future<PredictionResult> predictImage(XFile imageFile) async {
    final baseUrl = await getBaseUrl();
    final uri = Uri.parse('$baseUrl/api/predict');

    final request = http.MultipartRequest('POST', uri);
    final bytes = await imageFile.readAsBytes();
    final multipartFile = http.MultipartFile.fromBytes(
      'file',
      bytes,
      filename: imageFile.name,
    );
    request.files.add(multipartFile);

    final streamedResponse = await request.send().timeout(const Duration(seconds: 45));
    final response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      if (data['success'] == true) {
        return PredictionResult.fromJson(data);
      } else {
        throw Exception(data['error'] ?? 'Prediction failed on backend.');
      }
    } else {
      throw Exception('Server error (${response.statusCode}): ${response.body}');
    }
  }

  // 3. Predict Video Crowd Density Time Series
  static Future<VideoPredictionResult> predictVideo(XFile videoFile) async {
    final baseUrl = await getBaseUrl();
    final uri = Uri.parse('$baseUrl/api/predict-video');

    final request = http.MultipartRequest('POST', uri);
    final bytes = await videoFile.readAsBytes();
    final multipartFile = http.MultipartFile.fromBytes(
      'file',
      bytes,
      filename: videoFile.name,
    );
    request.files.add(multipartFile);

    final streamedResponse = await request.send().timeout(const Duration(seconds: 90));
    final response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return VideoPredictionResult.fromJson(data);
    } else {
      throw Exception('Video analysis failed (${response.statusCode}): ${response.body}');
    }
  }

  // 4. Calculate ROI Zone Count
  static Future<Map<String, dynamic>> calculateZoneCount(
      int analysisId, List<int> bboxPixels) async {
    final baseUrl = await getBaseUrl();
    final uri = Uri.parse('$baseUrl/api/zone-count');

    final response = await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'analysis_id': analysisId,
        'bbox': bboxPixels,
      }),
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Zone count error (${response.statusCode})');
    }
  }

  // 5. Fetch Scan History
  static Future<List<dynamic>> fetchHistory() async {
    final baseUrl = await getBaseUrl();
    final uri = Uri.parse('$baseUrl/api/history');

    final response = await http.get(uri).timeout(const Duration(seconds: 10));
    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return data['history'] ?? [];
    }
    return [];
  }

  // 6. Clear Scan History
  static Future<bool> clearHistory() async {
    final baseUrl = await getBaseUrl();
    final uri = Uri.parse('$baseUrl/api/history/clear');

    final response = await http.post(uri).timeout(const Duration(seconds: 10));
    return response.statusCode == 200;
  }
}
