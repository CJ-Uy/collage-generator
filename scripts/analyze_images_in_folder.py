import os
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import numpy as np
import json
import multiprocessing
from scripts.get_file_date import get_file_date


def save_to_json(data, output_file):
	"""Saves a dictionary to a JSON file."""
	try:
		with open(output_file, "w") as f:
			json.dump(data, f, indent=4, sort_keys=True)
		print(f"\nSuccessfully saved image data to {output_file}")
	except Exception as e:
		print(f"\nError saving data to JSON file. Reason: {e}")


def analyze_image(file_path):
	"""Analyzes a single image and returns its information."""
	try:
		filename = os.path.basename(file_path)
		with Image.open(file_path) as img:
			width, height = img.size
			area = width * height
			rgb_img = img.convert("RGB")
			np_array = np.array(rgb_img)
			avg_color = np.mean(np_array, axis=(0, 1))
			avg_rgb = tuple(int(round(c)) for c in avg_color)
			date_info = get_file_date(file_path)

			return filename, {
				"dimensions": f"{width}x{height}",
				"area": area,
				"average_rgb": avg_rgb,
				"date_taken": date_info,
			}
	except Exception as e:
		print(f"  [ERROR] Could not process {filename}. Reason: {e}")
		return None, None


def analyze_images_in_folder(folder_path):
	OUTPUT_JSON_FILE = "IMAGES_INFO.json"
	IMAGE_INFO = Path(OUTPUT_JSON_FILE)

	if IMAGE_INFO.exists():
		print(
			"IMAGES_INFO.json already exists! If you want to rerun analysis please delete it first!"
		)
	else:
		print(f"Scanning folder: {folder_path}")
		supported_formats = (".jpg", ".jpeg", ".png")

		image_paths = [
			os.path.join(folder_path, fn)
			for fn in os.listdir(folder_path)
			if fn.lower().endswith(supported_formats)
		]

		images_info = {}

		# Use multiprocessing to analyze images in parallel
		with multiprocessing.Pool() as pool:
			with tqdm(total=len(image_paths), desc="Analyzing Images") as pbar:
				for filename, info in pool.imap_unordered(analyze_image, image_paths):
					if filename:
						images_info[filename] = info
					pbar.update()

		print(f"Generating a {OUTPUT_JSON_FILE}")
		save_to_json(images_info, OUTPUT_JSON_FILE)
