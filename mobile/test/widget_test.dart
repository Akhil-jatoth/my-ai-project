import 'package:flutter_test/flutter_test.dart';
import 'package:crowdsense/main.dart';

void main() {
  testWidgets('CrowdSense app boots and displays splash screen', (WidgetTester tester) async {
    await tester.pumpWidget(const CrowdSenseApp());
    expect(find.text('CrowdSense'), findsOneWidget);
    expect(find.text('AI Deep Crowd Density Sensing'), findsOneWidget);
    
    // Drain animation timers
    await tester.pump(const Duration(seconds: 3));
  });
}
