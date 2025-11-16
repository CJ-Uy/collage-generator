"""Image loading and preprocessing utilities."""

import os
import json
from PIL import Image


def load_images_info(json_file="IMAGES_INFO.json"):
	"""Load preprocessed image information from JSON file."""
	with open(json_file, "r") as f:
		return json.load(f)


def load_and_scale_image(image_path, scale_factor=1.0):
	"""
	Load an image and optionally scale it down.

	Args:
	    image_path: Path to the image file
	    scale_factor: Factor to scale the image (1.0 = no scaling)

	Returns:
	    PIL Image object, or None if loading fails
	"""
	try:
		img = Image.open(image_path)

		if scale_factor < 1.0:
			original_width, original_height = img.size
			new_width = max(1, int(original_width * scale_factor))
			new_height = max(1, int(original_height * scale_factor))
			img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

		return img
	except Exception as e:
		print(f"\n[ERROR] Could not load {image_path}: {e}")
		return None
