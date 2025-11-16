"""Target image analysis utilities."""

import numpy as np
from PIL import Image


def analyze_target_image(target_path, cell_size=50):
	"""
	Analyze target image and extract average RGB for each grid cell.

	Args:
	    target_path: Path to target image
	    cell_size: Size of each grid cell in pixels

	Returns:
	    dict with 'grid' (RGB values per cell), 'image', 'dimensions'
	"""
	img = Image.open(target_path).convert("RGB")
	width, height = img.size
	np_img = np.array(img)

	# Calculate grid dimensions
	grid_cols = width // cell_size
	grid_rows = height // cell_size

	# Extract average color for each cell
	grid = {}
	for row in range(grid_rows):
		for col in range(grid_cols):
			y_start = row * cell_size
			y_end = (row + 1) * cell_size
			x_start = col * cell_size
			x_end = (col + 1) * cell_size

			cell_data = np_img[y_start:y_end, x_start:x_end]
			avg_color = np.mean(cell_data, axis=(0, 1))
			grid[(row, col)] = tuple(int(c) for c in avg_color)

	return {
		"grid": grid,
		"image": img,
		"dimensions": (width, height),
		"grid_size": (grid_rows, grid_cols),
		"cell_size": cell_size,
	}


def calculate_optimal_cell_size(images_info, percentile=10):
	"""
	Calculate optimal cell size based on image dimensions.

	Args:
	    images_info: Dict of image info from JSON
	    percentile: Use this percentile of smallest dimension (default 10 = smallest 10%)

	Returns:
	    Recommended cell size in pixels
	"""
	# Collect all minimum dimensions (smaller of width/height for each image)
	min_dimensions = []

	for info in images_info.values():
		dims = info["dimensions"]
		width, height = map(int, dims.split("x"))
		min_dim = min(width, height)
		min_dimensions.append(min_dim)

	if not min_dimensions:
		return 100  # Default fallback

	# Sort and get the percentile value
	min_dimensions.sort()
	index = max(0, int(len(min_dimensions) * percentile / 100) - 1)
	percentile_value = min_dimensions[index]

	# Use a fraction of this size to allow some overlap room
	# Using ~40% gives good balance between size and coverage
	cell_size = max(20, int(percentile_value * 0.4))

	return cell_size
