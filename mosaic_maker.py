"""
Mosaic Maker - Create a photo mosaic from your screenshots

This script creates a mosaic that looks like your target image,
composed of all your screenshot images with intelligent color matching.
"""

from scripts.analyze_images_in_folder import analyze_images_in_folder
from scripts.assemble_mosaic import assemble_mosaic

# ==================== CONFIGURATION ====================

# Target image - the image you want to recreate as a mosaic
TARGET_IMAGE = "target.jpg"

# Folder containing your screenshots/photos to use in the mosaic
IMAGE_FOLDER = "screenshots"

# Date filtering (OPTIONAL - set to None to include all images)
# Format: "YYYY-MM-DD"
START_DATE = None  # Example: "2023-02-10" - only include images from this date onward
END_DATE = None  # Example: "2023-11-07" - only include images up to this date

# Target image overlay (0.0 to 1.0)
# 0.0 = no overlay (just the mosaic)
# 0.3 = subtle overlay (recommended - helps see original image through mosaic)
# 1.0 = full overlay (target image fully visible)
OVERLAY_OPACITY = 0.3

# Output filename
OUTPUT_FILE = "mosaic_output.png"

# ======================================================

if __name__ == "__main__":
	print("=" * 60)
	print("MOSAIC MAKER")
	print("=" * 60)

	# Step 1: Analyze all images in folder (creates IMAGES_INFO.json)
	analyze_images_in_folder(IMAGE_FOLDER)

	# Step 2: Create the mosaic
	# Scale factor and canvas size are automatically calculated
	# to fit all images (after date filtering) efficiently
	assemble_mosaic(
		TARGET_IMAGE,
		IMAGE_FOLDER=IMAGE_FOLDER,
		output_file=OUTPUT_FILE,
		start_date=START_DATE,
		end_date=END_DATE,
		overlay_opacity=OVERLAY_OPACITY,
		# Advanced settings:
		scale_factor=None,  # Auto-calculated
		max_canvas_size=50000,  # Max dimension in pixels
		use_all_images=True,  # Always use all images
		tight_packing=True,  # Tight packing for better fit
		allow_duplicates=False,  # Each image used once
		allow_overlaps=False,  # No overlapping images
	)

	print("\n" + "=" * 60)
	print(f"✓ Mosaic saved to: {OUTPUT_FILE}")
	print("=" * 60)
