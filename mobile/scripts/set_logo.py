"""
CrowdSense — App Icon & Logo Switcher
Allows you to switch the mobile app launcher icon and in-app logo to any of the 5 designs:
    python scripts/set_logo.py 1   # Radar Iris & AI Vision Aperture
    python scripts/set_logo.py 2   # HexaPulse Biometric Grid
    python scripts/set_logo.py 3   # Minimalist CS Tech Monogram
    python scripts/set_logo.py 4   # Thermal Waveform Beacon
    python scripts/set_logo.py 5   # Ultra-Modern 3D Neon Hologram
"""

import sys
import os
import shutil
from PIL import Image

LOGO_MAP = {
    "1": "logo_1_radar_iris.png",
    "2": "logo_2_hexapulse.png",
    "3": "logo_3_cs_monogram.png",
    "4": "logo_4_thermal_beacon.png",
    "5": "logo_5_neon_3d.png",
}

def set_app_logo(choice="5"):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filename = LOGO_MAP.get(str(choice))
    if not filename:
        print(f"Error: Invalid choice '{choice}'. Choose 1, 2, 3, 4, or 5.")
        return

    source_path = os.path.join(base_dir, "assets", "logos", filename)
    if not os.path.exists(source_path):
        print(f"Error: Logo file not found: {source_path}")
        return

    # Update app logo and icon
    img = Image.open(source_path)
    img.save(os.path.join(base_dir, "assets", "images", "logo.png"))
    img.resize((512, 512), Image.Resampling.LANCZOS).save(os.path.join(base_dir, "assets", "icons", "app_icon.png"))

    # Update Android mipmap launcher icons
    res_map = {
        'mipmap-xxxhdpi': 192,
        'mipmap-xxhdpi': 144,
        'mipmap-xhdpi': 96,
        'mipmap-hdpi': 72,
        'mipmap-mdpi': 48,
    }
    for folder, px in res_map.items():
        p = os.path.join(base_dir, 'android', 'app', 'src', 'main', 'res', folder)
        if os.path.exists(p):
            icon_resized = img.resize((px, px), Image.Resampling.LANCZOS)
            icon_resized.save(os.path.join(p, 'ic_launcher.png'), 'PNG')

    print(f"SUCCESS: Set CrowdSense app logo to Option {choice} ({filename}) across all resolutions!")

if __name__ == "__main__":
    choice = sys.argv[1] if len(sys.argv) > 1 else "5"
    set_app_logo(choice)

