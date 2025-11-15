import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime


def load_images_info(json_file="IMAGES_INFO.json"):
	"""Load preprocessed image information from JSON file."""
	with open(json_file, "r") as f:
		return json.load(f)


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


def filter_images_by_date(images_info, start_date=None, end_date=None):
	"""
	Filter images by date range.

	Args:
	    images_info: Dict of image info from JSON
	    start_date: Start date as string (YYYY-MM-DD) or datetime object, or None for no lower bound
	    end_date: End date as string (YYYY-MM-DD) or datetime object, or None for no upper bound

	Returns:
	    Filtered dict of images within the date range
	"""
	# If no date filters, return all images
	if start_date is None and end_date is None:
		return images_info

	# Parse date strings to datetime objects if needed
	if start_date is not None and isinstance(start_date, str):
		start_date = datetime.strptime(start_date, "%Y-%m-%d")
	if end_date is not None and isinstance(end_date, str):
		end_date = datetime.strptime(end_date, "%Y-%m-%d")

	filtered = {}
	for filename, info in images_info.items():
		date_str = info.get("date_taken")
		if not date_str:
			# Skip images without date information
			continue

		try:
			# Parse the date_taken field (format: "YYYY-MM-DD HH:MM:SS")
			img_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
		except (ValueError, TypeError):
			# Skip images with invalid date format
			continue

		# Check if date is within range
		if start_date is not None and img_date < start_date:
			continue
		if end_date is not None and img_date > end_date:
			continue

		filtered[filename] = info

	return filtered


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


def color_distance(c1, c2):
	"""Calculate Euclidean distance between two RGB colors."""
	return np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))


def find_best_match(target_color, images_info, used_images):
	"""
	Find the best matching image for a target color.

	Args:
	    target_color: RGB tuple
	    images_info: Dict of image info from JSON
	    used_images: Set of already used image filenames

	Returns:
	    Filename of best matching unused image, or None
	"""
	best_match = None
	best_distance = float("inf")

	for filename, info in images_info.items():
		if filename in used_images:
			continue

		img_color = info["average_rgb"]
		distance = color_distance(target_color, img_color)

		if distance < best_distance:
			best_distance = distance
			best_match = filename

	return best_match


def get_candidate_positions(placed_rects, canvas_width, canvas_height, margin=2, max_candidates=200):
	"""
	Generate candidate positions for next image placement based on edges of placed images.

	Args:
	    placed_rects: List of (x, y, width, height, filename) tuples for placed images
	    canvas_width: Width of canvas
	    canvas_height: Height of canvas
	    margin: Minimum spacing between images (default 5px)
	    max_candidates: Maximum candidates to return (limits computation)

	Returns:
	    List of (x, y, distance_from_center) tuples sorted by distance from center
	"""
	if not placed_rects:
		# First image: place at center
		center_x = canvas_width // 2
		center_y = canvas_height // 2
		return [(center_x, center_y, 0)]

	candidates = []
	center_x = canvas_width // 2
	center_y = canvas_height // 2

	# Use all placed images for candidate generation
	# Generate positions around edges of ALL placed images
	for x, y, w, h, _ in placed_rects:
		# Only add 2 positions per edge (reduce from 3)
		# Right edge
		candidates.append((x + w + margin, y + h // 2))

		# Left edge
		candidates.append((x - margin, y + h // 2))

		# Bottom edge
		candidates.append((x + w // 2, y + h + margin))

		# Top edge
		candidates.append((x + w // 2, y - margin))

	# Calculate distance from center
	candidates_with_dist = []
	for x, y in candidates:
		dist = (x - center_x)**2 + (y - center_y)**2  # Use squared distance (faster)
		candidates_with_dist.append((x, y, dist))

	# Sort and return only top candidates
	candidates_with_dist.sort(key=lambda item: item[2])

	return candidates_with_dist[:max_candidates]


def check_overlap_fast(x1, y1, w1, h1, x2, y2, w2, h2):
	"""Fast overlap check for two rectangles."""
	return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)


def find_placement_position(img_width, img_height, placed_rects, canvas_width, canvas_height):
	"""
	Find a valid position to place an image without overlaps.

	Args:
	    img_width: Width of image to place
	    img_height: Height of image to place
	    placed_rects: List of (x, y, w, h, filename) tuples
	    canvas_width: Canvas width
	    canvas_height: Canvas height

	Returns:
	    (x, y) tuple for top-left position, or None if no valid position found
	"""
	candidates = get_candidate_positions(placed_rects, canvas_width, canvas_height)

	# Try multiple alignment strategies for each candidate position
	for x, y, _ in candidates:
		# Try different ways to align the image at this position
		alignment_strategies = [
			(x, y),  # Top-left corner at position
			(x - img_width, y),  # Top-right corner at position (place to left)
			(x, y - img_height),  # Bottom-left corner at position (place above)
			(x - img_width, y - img_height),  # Bottom-right corner at position
			(x - img_width // 2, y - img_height // 2),  # Center at position
		]

		for px, py in alignment_strategies:
			# Check bounds
			if px < 0 or py < 0 or px + img_width > canvas_width or py + img_height > canvas_height:
				continue

			# Check overlap with ALL placed images
			overlap = False

			for placed_x, placed_y, placed_w, placed_h, _ in placed_rects:
				if check_overlap_fast(px, py, img_width, img_height, placed_x, placed_y, placed_w, placed_h):
					overlap = True
					break

			if not overlap:
				return (px, py)

	return None


def check_overlap(new_rect, placed_rects):
	"""
	Check if a new rectangle overlaps with any placed rectangles.

	Args:
	    new_rect: (x1, y1, x2, y2) tuple for new rectangle
	    placed_rects: List of (x1, y1, x2, y2) tuples for placed rectangles

	Returns:
	    True if overlap exists, False otherwise
	"""
	x1_new, y1_new, x2_new, y2_new = new_rect

	for x1, y1, x2, y2 in placed_rects:
		# Check if rectangles overlap
		if not (x2_new <= x1 or x1_new >= x2 or y2_new <= y1 or y1_new >= y2):
			return True

	return False


def assemble_mosaic(
	TARGET_FILENAME,
	IMAGE_FOLDER="screenshots",
	cell_size=None,
	output_file="mosaic_output.png",
	start_date=None,
	end_date=None,
	use_all_images=True,
	scale_factor=None,
	max_canvas_size=20000,
):
	"""
	Assemble a mosaic from differently-sized images to recreate a target image.

	Args:
	    TARGET_FILENAME: Path to target image
	    IMAGE_FOLDER: Folder containing source images
	    cell_size: Size of grid cells for target analysis. If None, automatically calculates based on image count
	    output_file: Output filename for the mosaic
	    start_date: Start date for filtering images (YYYY-MM-DD format), None for no lower bound
	    end_date: End date for filtering images (YYYY-MM-DD format), None for no upper bound
	    use_all_images: If True, sizes the canvas to fit all images (default True)
	    scale_factor: Scale down images by this factor. If None, auto-calculates to keep within max_canvas_size
	    max_canvas_size: Maximum dimension (width or height) for output canvas in pixels (default 20000 = ~400MB PNG)
	"""
	print("\n=== MOSAIC ASSEMBLY ===")
	print(f"Target image: {TARGET_FILENAME}")
	print(f"Max canvas size: {max_canvas_size:,} px per dimension")

	# Load preprocessed image data
	print("\nLoading image information...")
	images_info = load_images_info()
	print(f"Loaded info for {len(images_info)} images")

	# Filter images by date range if specified
	if start_date is not None or end_date is not None:
		print("\nApplying date filter...")
		if start_date:
			print(f"  Start date: {start_date}")
		if end_date:
			print(f"  End date: {end_date}")
		images_info = filter_images_by_date(images_info, start_date, end_date)
		print(f"  Filtered to {len(images_info)} images within date range")

	num_images = len(images_info)

	# Analyze target image to get aspect ratio
	target_img = Image.open(TARGET_FILENAME)
	target_width_orig, target_height_orig = target_img.size
	aspect_ratio = target_width_orig / target_height_orig
	print(f"\nTarget image aspect ratio: {aspect_ratio:.2f}")

	# Calculate total area of all images
	if use_all_images:
		print("\nCalculating optimal canvas size...")
		total_area = sum(info["area"] for info in images_info.values())
		print(f"  Total image area: {total_area:,} px²")

		# If scale_factor not provided, calculate it based on max_canvas_size
		if scale_factor is None:
			# Start with unscaled calculation
			# Add 200% buffer for imperfect packing efficiency to ensure ALL images fit
			ideal_area = int(total_area * 3.0)

			# Calculate ideal canvas dimensions
			ideal_height = int(np.sqrt(ideal_area / aspect_ratio))
			ideal_width = int(ideal_height * aspect_ratio)

			# Check if either dimension exceeds max_canvas_size
			max_dim = max(ideal_width, ideal_height)

			if max_dim > max_canvas_size:
				# Calculate scale_factor needed to fit within max_canvas_size
				scale_factor = max_canvas_size / max_dim
				print(f"  Ideal canvas would be {ideal_width}x{ideal_height}")
				print(f"  Auto-scaling to fit within {max_canvas_size}px limit")
				print(f"  Calculated scale factor: {scale_factor * 100:.1f}%")
			else:
				# No scaling needed
				scale_factor = 1.0
				print("  Canvas fits within limit, no scaling needed")
		else:
			print(f"  Using user-specified scale factor: {scale_factor * 100:.0f}%")

		# Apply scaling to total area
		scaled_total_area = int(total_area * (scale_factor**2))
		target_area = int(scaled_total_area * 3.0)
		print(f"  Scaled image area: {scaled_total_area:,} px²")
		print(f"  Target canvas area (with packing buffer): {target_area:,} px²")

		# Calculate final canvas dimensions
		canvas_height = int(np.sqrt(target_area / aspect_ratio))
		canvas_width = int(canvas_height * aspect_ratio)

		# Ensure we don't exceed max_canvas_size (safety check)
		if max(canvas_width, canvas_height) > max_canvas_size:
			adjustment = max_canvas_size / max(canvas_width, canvas_height)
			canvas_width = int(canvas_width * adjustment)
			canvas_height = int(canvas_height * adjustment)
			print("  Applied final adjustment to ensure size limit")

		print(
			f"Final canvas size: {canvas_width}x{canvas_height} ({canvas_width * canvas_height:,} px²)"
		)

		# Estimate file size (rough approximation for PNG)
		estimated_mb = (canvas_width * canvas_height * 4) / (
			1024 * 1024
		)  # RGBA = 4 bytes per pixel
		print(f"Estimated file size: ~{estimated_mb:.0f} MB")
	else:
		canvas_width = target_width_orig * 2
		canvas_height = target_height_orig * 2
		if scale_factor is None:
			scale_factor = 1.0

	# Calculate optimal cell size if not provided
	if cell_size is None:
		if use_all_images:
			# Create MORE cells than images to give flexibility in placement
			# Aim for 2x the number of images to ensure we can place everything
			total_cells_needed = int(num_images * 2.0)
			cell_area = (canvas_width * canvas_height) / total_cells_needed
			cell_size = max(20, int(np.sqrt(cell_area)))
			print(f"\nCalculated cell size to accommodate all images: {cell_size}px")
			print(
				f"  This will create ~{total_cells_needed} cells for {num_images} images"
			)
		else:
			cell_size = calculate_optimal_cell_size(images_info)
			print(f"\nCalculated cell size: {cell_size}px")
	else:
		print(f"\nUsing specified cell size: {cell_size}px")

	# Analyze target image with calculated cell size
	print("\nAnalyzing target image grid...")
	target_data = analyze_target_image(TARGET_FILENAME, cell_size)
	grid = target_data["grid"]

	# Scale grid to canvas size
	grid_rows = canvas_height // cell_size
	grid_cols = canvas_width // cell_size

	# Regenerate grid at canvas scale
	print("Resizing target image to canvas scale...")
	target_img_resized = target_img.resize(
		(canvas_width, canvas_height), Image.Resampling.LANCZOS
	)
	np_img = np.array(target_img_resized.convert("RGB"))

	print("Analyzing color grid...")
	grid = {}
	for row in tqdm(range(grid_rows), desc="Analyzing grid"):
		for col in range(grid_cols):
			y_start = row * cell_size
			y_end = min((row + 1) * cell_size, canvas_height)
			x_start = col * cell_size
			x_end = min((col + 1) * cell_size, canvas_width)

			cell_data = np_img[y_start:y_end, x_start:x_end]
			if cell_data.size > 0:
				avg_color = np.mean(cell_data, axis=(0, 1))
				grid[(row, col)] = tuple(int(c) for c in avg_color)

	print(f"Grid: {grid_rows} rows x {grid_cols} cols = {len(grid)} cells")
	print(f"Images to place: {num_images}")

	# Create output canvas
	canvas = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))

	# Track used images and placed rectangles (x, y, width, height, filename)
	used_images = set()
	placed_rects = []

	# Place images using organic packing from center outward
	print("\nPlacing images organically from center outward...")

	# Create a grid-based sampling of the target for color matching
	# We'll sample colors from the grid to match images to regions
	color_samples = []
	for row in range(grid_rows):
		for col in range(grid_cols):
			if (row, col) in grid:
				x = col * cell_size + cell_size // 2
				y = row * cell_size + cell_size // 2
				color = grid[(row, col)]
				# Calculate distance from center for priority
				dist = np.sqrt((x - canvas_width//2)**2 + (y - canvas_height//2)**2)
				color_samples.append((x, y, color, dist))

	# Sort by distance from center (process center first)
	color_samples.sort(key=lambda item: item[3])

	# Place images
	failed_placements = 0
	placement_attempts = 0

	for img_idx in tqdm(range(num_images), desc="Assembling"):
		placement_attempts += 1
		if len(used_images) >= num_images:
			break

		# Find target color from nearest unprocessed sample
		target_color = None
		if img_idx < len(color_samples):
			_, _, target_color, _ = color_samples[img_idx]
		else:
			# Use center color as fallback
			center_row, center_col = grid_rows // 2, grid_cols // 2
			target_color = grid.get((center_row, center_col), (128, 128, 128))

		# Find best matching unused image (limit search for speed)
		best_match = find_best_match(target_color, images_info, used_images)

		if best_match is None:
			# No more unused images available
			failed_placements += 1
			continue

		# Load the matched image
		img_path = os.path.join(IMAGE_FOLDER, best_match)
		try:
			img = Image.open(img_path)

			# Scale down the image if scale_factor < 1.0
			if scale_factor < 1.0:
				original_width, original_height = img.size
				new_width = max(1, int(original_width * scale_factor))
				new_height = max(1, int(original_height * scale_factor))
				img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
		except Exception as e:
			print(f"\n[ERROR] Could not load {best_match}: {e}")
			failed_placements += 1
			continue

		# Find placement position using organic packing
		img_width, img_height = img.size
		position = find_placement_position(img_width, img_height, placed_rects, canvas_width, canvas_height)

		if position is None:
			# Could not find valid position - skip this image
			failed_placements += 1
			continue

		paste_x, paste_y = position

		# Paste the image onto canvas (handle transparency if present)
		if img.mode == "RGBA":
			canvas.paste(img, (paste_x, paste_y), img)
		else:
			# Convert to RGBA to maintain consistency
			img_rgba = img.convert("RGBA")
			canvas.paste(img_rgba, (paste_x, paste_y), img_rgba)

		# Mark image as used and record placement
		used_images.add(best_match)
		placed_rects.append((paste_x, paste_y, img_width, img_height, best_match))
		failed_placements = 0  # Reset failure counter on success

	print(f"\nPlacement attempts: {placement_attempts}")
	print(f"Successfully placed: {len(used_images)} images")
	print(f"Failed placements: {failed_placements}")
	print(f"Used {len(used_images)} unique images out of {num_images} available")

	usage_percentage = (len(used_images) / num_images * 100) if num_images > 0 else 0
	print(f"Image usage: {usage_percentage:.1f}%")

	if len(used_images) < num_images:
		unused_count = num_images - len(used_images)
		print(f"\nNote: {unused_count} images were not placed due to space constraints")
		print("To use more images, try:")
		print("  - Increasing max_canvas_size")
		print("  - Decreasing scale_factor")

	# Crop canvas to actual content (optional - remove whitespace)
	# For now, we'll keep the full canvas to show all placed images

	# Save the result
	print(f"\nSaving mosaic to {output_file}...")
	canvas.save(output_file)
	print("✓ Mosaic saved successfully!")

	return canvas
