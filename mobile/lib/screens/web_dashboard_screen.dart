import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:webview_flutter/webview_flutter.dart';
import 'package:webview_flutter_android/webview_flutter_android.dart';
import '../theme/app_theme.dart';

class WebDashboardScreen extends StatefulWidget {
  final String? initialUrl;

  const WebDashboardScreen({super.key, this.initialUrl});

  @override
  State<WebDashboardScreen> createState() => _WebDashboardScreenState();
}

class _WebDashboardScreenState extends State<WebDashboardScreen> {
  late final WebViewController _controller;
  final TextEditingController _urlInputController = TextEditingController();

  String _currentUrl = "http://10.235.164.46:5000";
  bool _isLoading = true;
  double _progress = 0.0;
  bool _hasError = false;
  String _errorMessage = "";
  bool _isDesktopView = false;
  bool _showUrlBar = false;

  static const String _desktopUA =
      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36";
  static const String _mobileUA =
      "Mozilla/5.0 (Linux; Android 14; Mobile) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Mobile Safari/537.36";

  @override
  void initState() {
    super.initState();
    _initWebView();
    _loadSavedUrl();
  }

  void _initWebView() {
    final PlatformWebViewControllerCreationParams params;
    if (WebViewPlatform.instance is AndroidWebViewPlatform) {
      params = AndroidWebViewControllerCreationParams();
    } else {
      params = const PlatformWebViewControllerCreationParams();
    }

    _controller = WebViewController.fromPlatformCreationParams(params)
      ..setJavaScriptMode(JavaScriptMode.unrestricted)
      ..setBackgroundColor(const Color(0xFF0F172A))
      ..setNavigationDelegate(
        NavigationDelegate(
          onProgress: (int progress) {
            if (mounted) {
              setState(() {
                _progress = progress / 100.0;
              });
            }
          },
          onPageStarted: (String url) {
            if (mounted) {
              setState(() {
                _isLoading = true;
                _hasError = false;
                _currentUrl = url;
                _urlInputController.text = url;
              });
            }
          },
          onPageFinished: (String url) {
            if (mounted) {
              setState(() {
                _isLoading = false;
              });
            }
          },
          onWebResourceError: (WebResourceError error) {
            if (error.isForMainFrame ?? true) {
              if (mounted) {
                setState(() {
                  _hasError = true;
                  _errorMessage = error.description;
                  _isLoading = false;
                });
              }
            }
          },
        ),
      );

    if (_controller.platform is AndroidWebViewController) {
      final androidController = _controller.platform as AndroidWebViewController;
      AndroidWebViewController.enableDebugging(true);
      androidController.setMediaPlaybackRequiresUserGesture(false);
      androidController.setOnPlatformPermissionRequest((request) {
        request.grant();
      });
    }
  }

  Future<void> _loadSavedUrl() async {
    final prefs = await SharedPreferences.getInstance();
    final saved = widget.initialUrl ??
        prefs.getString('backend_url') ??
        'http://10.235.164.46:5000';

    setState(() {
      _currentUrl = saved;
      _urlInputController.text = saved;
    });

    _loadUrl(saved);
  }

  Future<void> _loadUrl(String url) async {
    String formattedUrl = url.trim();
    if (!formattedUrl.startsWith('http://') && !formattedUrl.startsWith('https://')) {
      formattedUrl = 'http://$formattedUrl';
    }

    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('backend_url', formattedUrl);

    try {
      await _controller.loadRequest(Uri.parse(formattedUrl));
    } catch (e) {
      setState(() {
        _hasError = true;
        _errorMessage = e.toString();
        _isLoading = false;
      });
    }
  }

  void _toggleViewMode() {
    setState(() {
      _isDesktopView = !_isDesktopView;
    });
    _controller.setUserAgent(_isDesktopView ? _desktopUA : _mobileUA);
    _controller.reload();

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(
          _isDesktopView
              ? "💻 Switched to Laptop (Desktop) Layout"
              : "📱 Switched to Mobile Responsive Layout",
        ),
        duration: const Duration(seconds: 2),
        backgroundColor: AppTheme.primary,
      ),
    );
  }

  @override
  void dispose() {
    _urlInputController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return PopScope(
      canPop: false,
      onPopInvokedWithResult: (didPop, result) async {
        if (didPop) return;
        if (await _controller.canGoBack()) {
          await _controller.goBack();
        } else {
          if (context.mounted) {
            Navigator.of(context).maybePop();
          }
        }
      },
      child: Scaffold(
        backgroundColor: const Color(0xFF0F172A),
        appBar: AppBar(
          backgroundColor: const Color(0xFF1E293B),
          elevation: 2,
          leading: Padding(
            padding: const EdgeInsets.all(8.0),
            child: Container(
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(8),
                gradient: const LinearGradient(
                  colors: [Color(0xFF4F46E5), Color(0xFF7C3AED)],
                ),
              ),
              child: const Icon(Icons.groups_rounded, color: Colors.white, size: 22),
            ),
          ),
          titleSpacing: 0,
          title: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'CrowdSense',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                  letterSpacing: -0.5,
                  color: Colors.white,
                ),
              ),
              Text(
                _currentUrl.replaceFirst(RegExp(r'^https?://'), ''),
                style: const TextStyle(
                  fontSize: 11,
                  color: Color(0xFF94A3B8),
                  fontWeight: FontWeight.w400,
                ),
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
              ),
            ],
          ),
          actions: [
            // Mode toggle: Laptop / Mobile
            IconButton(
              icon: Icon(
                _isDesktopView ? Icons.laptop_chromebook : Icons.smartphone,
                color: _isDesktopView ? AppTheme.accentCyan : Colors.white70,
              ),
              tooltip: _isDesktopView ? "Laptop Layout Active" : "Mobile Layout Active",
              onPressed: _toggleViewMode,
            ),
            // URL Server Settings
            IconButton(
              icon: Icon(
                _showUrlBar ? Icons.close : Icons.link,
                color: _showUrlBar ? AppTheme.highRisk : Colors.white70,
              ),
              tooltip: "Change Server URL",
              onPressed: () {
                setState(() {
                  _showUrlBar = !_showUrlBar;
                });
              },
            ),
            // Refresh
            IconButton(
              icon: const Icon(Icons.refresh, color: Colors.white70),
              tooltip: "Reload Page",
              onPressed: () {
                setState(() {
                  _hasError = false;
                });
                _controller.reload();
              },
            ),
          ],
          bottom: _isLoading
              ? PreferredSize(
                  preferredSize: const Size.fromHeight(3.0),
                  child: LinearProgressIndicator(
                    value: _progress > 0 ? _progress : null,
                    backgroundColor: Colors.transparent,
                    valueColor: const AlwaysStoppedAnimation<Color>(Color(0xFF4F46E5)),
                    minHeight: 3,
                  ),
                )
              : null,
        ),
        body: Column(
          children: [
            // Collapsible URL Input Bar
            if (_showUrlBar)
              Container(
                color: const Color(0xFF1E293B),
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                child: Row(
                  children: [
                    Expanded(
                      child: TextField(
                        controller: _urlInputController,
                        style: const TextStyle(color: Colors.white, fontSize: 13),
                        decoration: InputDecoration(
                          hintText: 'http://10.235.164.46:5000',
                          hintStyle: const TextStyle(color: Colors.white38),
                          isDense: true,
                          contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                          filled: true,
                          fillColor: const Color(0xFF0F172A),
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(8),
                            borderSide: const BorderSide(color: Color(0xFF334155)),
                          ),
                          focusedBorder: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(8),
                            borderSide: const BorderSide(color: Color(0xFF4F46E5)),
                          ),
                        ),
                        onSubmitted: (val) {
                          _loadUrl(val);
                          setState(() => _showUrlBar = false);
                        },
                      ),
                    ),
                    const SizedBox(width: 8),
                    ElevatedButton(
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0xFF4F46E5),
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                      ),
                      onPressed: () {
                        _loadUrl(_urlInputController.text);
                        setState(() => _showUrlBar = false);
                      },
                      child: const Text('Go', style: TextStyle(fontWeight: FontWeight.bold)),
                    ),
                  ],
                ),
              ),

            // Main Content: WebView or Connection Error
            Expanded(
              child: Stack(
                children: [
                  WebViewWidget(controller: _controller),
                  if (_hasError) _buildErrorScreen(),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildErrorScreen() {
    return Container(
      color: const Color(0xFF0F172A),
      width: double.infinity,
      padding: const EdgeInsets.all(24.0),
      child: Center(
        child: SingleChildScrollView(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: AppTheme.criticalRisk.withOpacity(0.12),
                  shape: BoxShape.circle,
                ),
                child: const Icon(
                  Icons.cloud_off_rounded,
                  size: 56,
                  color: AppTheme.criticalRisk,
                ),
              ),
              const SizedBox(height: 20),
              const Text(
                'Cannot Reach CrowdSense Server',
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 10),
              Text(
                'Attempted to load: $_currentUrl\n$_errorMessage',
                style: const TextStyle(fontSize: 13, color: Color(0xFF94A3B8)),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 24),

              // Connect to Laptop instructions
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: const Color(0xFF1E293B),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: const Color(0xFF334155)),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      '📱 How to connect your phone:',
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                        fontSize: 14,
                      ),
                    ),
                    const SizedBox(height: 8),
                    const Text(
                      '1. Make sure your laptop and phone are on the same Wi-Fi.\n'
                      '2. Run "python app.py" on your laptop.\n'
                      '3. Enter your laptop\'s IP address below and tap Connect.',
                      style: TextStyle(color: Color(0xFFCBD5E1), fontSize: 13, height: 1.4),
                    ),
                    const SizedBox(height: 14),
                    TextField(
                      controller: _urlInputController,
                      style: const TextStyle(color: Colors.white, fontSize: 14),
                      decoration: InputDecoration(
                        labelText: 'Server URL',
                        labelStyle: const TextStyle(color: Color(0xFF94A3B8)),
                        filled: true,
                        fillColor: const Color(0xFF0F172A),
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(8),
                          borderSide: const BorderSide(color: Color(0xFF475569)),
                        ),
                      ),
                    ),
                    const SizedBox(height: 12),
                    const Text(
                      'Quick Presets:',
                      style: TextStyle(color: Color(0xFF94A3B8), fontSize: 12),
                    ),
                    const SizedBox(height: 6),
                    Wrap(
                      spacing: 8,
                      runSpacing: 6,
                      children: [
                        _buildPresetChip('http://10.235.164.46:5000', 'Laptop Wi-Fi IP'),
                        _buildPresetChip('http://10.0.2.2:5000', 'Android Emulator'),
                        _buildPresetChip('http://127.0.0.1:5000', 'Localhost'),
                      ],
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 20),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton.icon(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF4F46E5),
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 14),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                  ),
                  icon: const Icon(Icons.refresh),
                  label: const Text(
                    'Connect & Open Interface',
                    style: TextStyle(fontSize: 15, fontWeight: FontWeight.bold),
                  ),
                  onPressed: () {
                    _loadUrl(_urlInputController.text);
                  },
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildPresetChip(String url, String label) {
    return ActionChip(
      backgroundColor: const Color(0xFF0F172A),
      label: Text(label, style: const TextStyle(color: AppTheme.accentCyan, fontSize: 12)),
      avatar: const Icon(Icons.flash_on, size: 14, color: AppTheme.accentCyan),
      side: const BorderSide(color: Color(0xFF334155)),
      onPressed: () {
        _urlInputController.text = url;
        _loadUrl(url);
      },
    );
  }
}
