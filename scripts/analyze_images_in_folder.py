import os
from tqdm import tqdm
from pathlib import Path

from PIL import Image
import numpy as np
import json

from scripts.get_file_date import get_file_date

def save_to_json(data, output_file):
	"""Saves a dictionary to a JSON file."""
	try:
		with open(output_file, "w") as f:
			json.dump(data, f, indent=4, sort_keys=True)
		print(f"\nSuccessfully saved image data to {output_file}")
	except Exception as e:
		print(f"\nError saving data to JSON file. Reason: {e}")

def analyze_images_in_folder(folder_path):
	OUTPUT_JSON_FILE = "IMAGES_INFO.json"
	IMAGE_INFO = Path(OUTPUT_JSON_FILE)

	if IMAGE_INFO.exists():
		print("IMAGE_INFO.json already exists! If you want to rerun analysis please delete it first!")
	else:
		print(f"Scanning folder: {folder_path}")
		supported_formats = (".jpg", ".jpeg", ".png")
		images_info = {}

		# Loop through all files in the directory
		for filename in tqdm(os.listdir(folder_path)):
			# Check if the file is an image
			if filename.lower().endswith(supported_formats):
				file_path = os.path.join(folder_path, filename)

				try:
					# Open the image file
					with Image.open(file_path) as img:
						# 1. Get Dimensions
						width, height = img.size

						# 2. Calculate Area
						area = width * height

						# 3. Get Average RGB Color
						# Convert image to RGB to handle different modes (like RGBA, P, L)
						rgb_img = img.convert("RGB")
						# Use numpy for fast calculation
						np_array = np.array(rgb_img)
						avg_color = np.mean(np_array, axis=(0, 1))
						# Convert numpy floats to a tuple of integers
						avg_rgb = tuple(int(round(c)) for c in avg_color)

						# 4. Get Metadata (Date Taken)
						date_info = get_file_date(file_path)

						# Store all the collected info in the dictionary
						images_info[filename] = {
							"dimensions": f"{width}x{height}",
							"area": area,
							"average_rgb": avg_rgb,
							"date_taken": date_info,
						}

				except Exception as e:
					print(f"  [ERROR] Could not process {filename}. Reason: {e}")
		
		print(f"Generating a {OUTPUT_JSON_FILE}")
		save_to_json(images_info, OUTPUT_JSON_FILE)
      