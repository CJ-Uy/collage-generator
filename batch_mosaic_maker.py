"""
Batch Mosaic Maker - Process multiple target images with date-based filtering

Place target images in 'targets/' folder with filenames: "YYYY-MM-DD to YYYY-MM-DD.jpg"
Mosaics will be saved to 'outputs/' folder as: "YYYY-MM-DD to YYYY-MM-DD Mosaic.png"
"""

from scripts.analyze_images_in_folder import analyze_images_in_folder
from scripts.batch_processor import process_targets_folder

# ==================== CONFIGURATION ====================

# Folder containing target images (name format: "YYYY-MM-DD to YYYY-MM-DD.jpg")
TARGETS_FOLDER = "targets"

# Folder containing your screenshots/photos to use in the mosaics
SCREENSHOTS_FOLDER = "screenshots"

# Folder where output mosaics will be saved
OUTPUTS_FOLDER = "outputs"

# Target image overlay (0.0 to 1.0)
# 0.0 = no overlay (just the mosaic)
# 0.3 = subtle overlay (recommended - helps see original image through mosaic)
# 1.0 = full overlay (target image fully visible)
OVERLAY_OPACITY = 0.3

# ======================================================

if __name__ == "__main__":
	print("=" * 70)
	print("BATCH MOSAIC MAKER")
	print("=" * 70)
	print("\nProcessing all target images in 'targets/' folder...")
	print("Target filenames should be: 'YYYY-MM-DD to YYYY-MM-DD.jpg'")
	print("Example: '2023-01-01 to 2023-12-31.jpg'")
	print("\nOutput mosaics will be saved to 'outputs/' folder")
	print("=" * 70)

	# Analyze images in folder (creates IMAGES_INFO.json if needed)
	analyze_images_in_folder(SCREENSHOTS_FOLDER)

	# Process all target images (verbose=False for cleaner output)
	process_targets_folder(
		targets_folder=TARGETS_FOLDER,
		screenshots_folder=SCREENSHOTS_FOLDER,
		outputs_folder=OUTPUTS_FOLDER,
		overlay_opacity=OVERLAY_OPACITY,
		verbose=False
	)
