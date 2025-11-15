from scripts.analyze_images_in_folder import analyze_images_in_folder
from scripts.assemble_mosaic import assemble_mosaic

TARGET_IMAGE = "target.jpg"
IMAGE_FOLDER = "screenshots"

analyze_images_in_folder(IMAGE_FOLDER)
# Use very small scale (2%) with tight packing to fit ALL images
assemble_mosaic(
	TARGET_IMAGE,
	scale_factor=0.02,
	max_canvas_size=50000,
	tight_packing=True,
	allow_overlaps=False,
	allow_duplicates=False,  # Set to True to allow image reuse while ensuring all images appear at least once
)
