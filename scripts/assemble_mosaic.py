import os
import json
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from scipy.spatial import cKDTree


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

def find_best_match(target_color, tree, image_filenames, used_images, k=10):
    """
    Find the best matching unused image for a target color using a k-d tree.

    Args:
        target_color: RGB tuple
        tree: cKDTree of image colors
        image_filenames: List of filenames corresponding to the tree data
        used_images: Set of already used image filenames
        k: Number of nearest neighbors to check

    Returns:
        Filename of best matching unused image, or None
    """
    # Query the tree for the k nearest neighbors
    distances, indices = tree.query(target_color, k=k)

    # Find the first unused image among the neighbors
    for index in indices:
        filename = image_filenames[index]
        if filename not in used_images:
            return filename

    return None

def get_spiral_order_cells(grid_rows, grid_cols):
    """
    Generate grid cell coordinates in spiral order from center outward.

    Args:
        grid_rows: Number of rows in grid
        grid_cols: Number of columns in grid

    Yields:
        (row, col) tuples in spiral order
    """
    visited = set()
    center_row, center_col = grid_rows // 2, grid_cols // 2

    # Start from center
    if 0 <= center_row < grid_rows and 0 <= center_col < grid_cols:
        yield (center_row, center_col)
        visited.add((center_row, center_col))

    # Spiral outward
    row, col = center_row, center_col
    dx, dy = 0, -1  # Start by moving up
    steps = 1
    turn_count = 0

    while len(visited) < grid_rows * grid_cols:
        for _ in range(steps):
            row, col = row + dy, col + dx  # Note: row uses dy, col uses dx
            if 0 <= row < grid_rows and 0 <= col < grid_cols and (row, col) not in visited:
                yield (row, col)
                visited.add((row, col))

        # Turn 90 degrees
        dx, dy = -dy, dx
        turn_count += 1

        if turn_count % 2 == 0:
            steps += 1

        # Safety check to prevent infinite loop
        if turn_count > grid_rows * grid_cols * 4:
            break

def find_placement_position_fast(img_width, img_height, occupancy_grid, grid_size, canvas_width, canvas_height):
    """
    Find a valid position to place an image using a fast occupancy grid with randomized offsets.

    Args:
        img_width: Width of image to place
        img_height: Height of image to place
        occupancy_grid: 2D numpy array marking occupied cells
        grid_size: Size of each grid cell in pixels
        canvas_width: Canvas width
        canvas_height: Canvas height

    Returns:
        (x, y) tuple for top-left position, or None if no valid position found
    """
    grid_rows, grid_cols = occupancy_grid.shape

    # Try positions in sequential order (left-to-right, top-to-bottom) for tight packing
    for row in range(grid_rows):
        for col in range(grid_cols):
            # Check if this grid cell and surrounding cells are free
            cells_needed_h = (img_height + grid_size - 1) // grid_size
            cells_needed_w = (img_width + grid_size - 1) // grid_size

            # Check if we have enough free cells
            if row + cells_needed_h > grid_rows or col + cells_needed_w > grid_cols:
                continue

            # Check if all needed cells are free
            region = occupancy_grid[row:row+cells_needed_h, col:col+cells_needed_w]
            if np.any(region):
                continue

            # Calculate pixel position - align to grid for tight packing (no random offset)
            px = col * grid_size
            py = row * grid_size

            # Ensure we stay in bounds
            px = max(0, min(px, canvas_width - img_width))
            py = max(0, min(py, canvas_height - img_height))

            # Check canvas bounds
            if px + img_width > canvas_width or py + img_height > canvas_height or px < 0 or py < 0:
                continue

            return (px, py)

    return None

def mark_occupied(occupancy_grid, x, y, width, height, grid_size):
    """
    Mark grid cells as occupied by an image.

    Args:
        occupancy_grid: 2D numpy array
        x, y: Top-left position of image
        width, height: Image dimensions
        grid_size: Size of each grid cell
    """
    grid_rows, grid_cols = occupancy_grid.shape

    # Calculate which cells are covered - exact footprint only
    start_col = x // grid_size
    start_row = y // grid_size
    end_col = min((x + width + grid_size - 1) // grid_size, grid_cols)
    end_row = min((y + height + grid_size - 1) // grid_size, grid_rows)

    # Mark cells as occupied
    occupancy_grid[start_row:end_row, start_col:end_col] = 1

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
	allow_overlaps=False,
	tight_packing=True,
	allow_duplicates=False,
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
	    allow_overlaps: If True, allows images to overlap slightly for better packing (default False)
	    tight_packing: If True, uses sequential placement for tighter packing with less empty space (default True)
	    allow_duplicates: If True, allows reusing images to fill all grid cells. All unique images will still be shown at least once (default False)
	"""
	print("\n=== MOSAIC ASSEMBLY ===")
	print(f"Target image: {TARGET_FILENAME}")
	if max_canvas_size:
		print(f"Max canvas size: {max_canvas_size:,} px per dimension")
	else:
		print("Max canvas size: Unlimited (auto-size to fit all images)")

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
	if num_images == 0:
		print("No images to process. Exiting.")
		return

	# Prepare data for k-d tree
	image_filenames = list(images_info.keys())
	image_colors = [images_info[fn]["average_rgb"] for fn in image_filenames]
	tree = cKDTree(image_colors)

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
			# With sequential tight packing, we can use a much smaller buffer
			if tight_packing:
				ideal_area = int(total_area * 1.5)  # 50% buffer for tight sequential packing
			else:
				ideal_area = int(total_area * 3.0)  # 200% buffer for randomized placement

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
		if tight_packing:
			target_area = int(scaled_total_area * 1.5)  # 50% buffer for tight sequential packing
		else:
			target_area = int(scaled_total_area * 3.0)  # 200% buffer for randomized placement
		print(f"  Scaled image area: {scaled_total_area:,} px²")
		print(f"  Target canvas area (with packing buffer): {target_area:,} px²")

		# Calculate final canvas dimensions
		canvas_height = int(np.sqrt(target_area / aspect_ratio))
		canvas_width = int(canvas_height * aspect_ratio)

		# Ensure we don't exceed max_canvas_size (safety check)
		if max_canvas_size and max(canvas_width, canvas_height) > max_canvas_size:
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
			# Create cells based on actual image count with extra space for placement flexibility
			if tight_packing:
				total_cells_needed = int(num_images * 1.2)  # Only 20% extra for tight packing
			else:
				total_cells_needed = int(num_images * 2.5)  # 150% extra cells for placement flexibility
			cell_area = (canvas_width * canvas_height) / total_cells_needed
			cell_size = max(10, int(np.sqrt(cell_area)))
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

	# Create occupancy grid for fast collision detection
	# Use smaller cells for more precise packing
	if tight_packing:
		occupancy_cell_size = max(5, cell_size // 12)  # Extra small cells for very tight packing
	else:
		occupancy_cell_size = max(15, cell_size // 8)  # Standard small cells
	occupancy_rows = (canvas_height + occupancy_cell_size - 1) // occupancy_cell_size
	occupancy_cols = (canvas_width + occupancy_cell_size - 1) // occupancy_cell_size
	occupancy_grid = np.zeros((occupancy_rows, occupancy_cols), dtype=np.uint8)

	print(f"Occupancy grid: {occupancy_rows}x{occupancy_cols} cells of {occupancy_cell_size}px")

	# Track placed images
	placed_images = []

	# Sort images by area (largest first) for better bin packing
	print("\nSorting images by size (largest first)...")
	images_by_size = sorted(
		images_info.items(),
		key=lambda x: x[1]["area"],
		reverse=True
	)

	# PHASE 1: Color-matched placement in spiral order from center
	print("\nPhase 1: Placing images with color matching...")
	used_images = set()
	placed_at_least_once = set()  # Track which images have been placed at least once

	# Get spiral order of grid cells (center outward)
	spiral_cells = list(get_spiral_order_cells(grid_rows, grid_cols))

	# Try to match each grid cell with best-matching image
	for row, col in tqdm(spiral_cells, desc="Color matching"):
		# Stop only if duplicates are disabled and we've used all images
		if not allow_duplicates and len(used_images) >= num_images:
			break

		# Get target color for this cell
		target_color = grid.get((row, col))
		if target_color is None:
			continue

		# Find best matching image
		if allow_duplicates:
			# When duplicates allowed, prefer unused images first, then allow reuse
			if len(placed_at_least_once) < num_images:
				# Still have images that haven't been placed once
				best_match = find_best_match(target_color, tree, image_filenames, placed_at_least_once, k=50)
			else:
				# All images placed at least once, now allow any best match
				best_match = find_best_match(target_color, tree, image_filenames, set(), k=50)
		else:
			# No duplicates - only use each image once
			best_match = find_best_match(target_color, tree, image_filenames, used_images, k=50)

		if best_match is None:
			continue

		# Load and scale the image
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
			continue

		img_width, img_height = img.size

		# Try to place near the target cell
		target_x = col * cell_size
		target_y = row * cell_size

		# Search in expanding radius around target position
		found_position = None
		for search_radius in range(0, max(grid_rows, grid_cols)):
			# Try positions in a square around target
			for dy in range(-search_radius, search_radius + 1):
				for dx in range(-search_radius, search_radius + 1):
					# Only check perimeter of square
					if search_radius > 0 and abs(dx) != search_radius and abs(dy) != search_radius:
						continue

					check_x = max(0, min(target_x + dx * cell_size, canvas_width - img_width))
					check_y = max(0, min(target_y + dy * cell_size, canvas_height - img_height))

					# Check if position is valid in occupancy grid
					start_col_occ = check_x // occupancy_cell_size
					start_row_occ = check_y // occupancy_cell_size
					cells_needed_h = (img_height + occupancy_cell_size - 1) // occupancy_cell_size
					cells_needed_w = (img_width + occupancy_cell_size - 1) // occupancy_cell_size

					if start_row_occ + cells_needed_h > occupancy_rows or start_col_occ + cells_needed_w > occupancy_cols:
						continue

					region = occupancy_grid[start_row_occ:start_row_occ+cells_needed_h, start_col_occ:start_col_occ+cells_needed_w]
					if not np.any(region):
						found_position = (check_x, check_y)
						break
				if found_position:
					break
			if found_position:
				break

		if found_position is None:
			continue

		paste_x, paste_y = found_position

		# Paste the image onto canvas
		if img.mode == "RGBA":
			canvas.paste(img, (paste_x, paste_y), img)
		else:
			img_rgba = img.convert("RGBA")
			canvas.paste(img_rgba, (paste_x, paste_y), img_rgba)

		# Mark cells as occupied
		mark_occupied(occupancy_grid, paste_x, paste_y, img_width, img_height, occupancy_cell_size)

		# Track as used
		if not allow_duplicates:
			used_images.add(best_match)
		placed_at_least_once.add(best_match)
		placed_images.append(best_match)

	print(f"Phase 1 complete: {len(placed_images)} images placed")
	print(f"  Unique images used: {len(placed_at_least_once)}")

	# PHASE 2: Place remaining images anywhere they fit (only if duplicates disabled)
	if not allow_duplicates:
		print("\nPhase 2: Placing remaining images in available space...")
		remaining_images = [fn for fn in image_filenames if fn not in placed_at_least_once]

		# Sort remaining by size (largest first for better packing)
		remaining_sorted = sorted(
			remaining_images,
			key=lambda fn: images_info[fn]["area"],
			reverse=True
		)

		for filename in tqdm(remaining_sorted, desc="Filling gaps"):
			# Load and scale the image
			img_path = os.path.join(IMAGE_FOLDER, filename)
			try:
				img = Image.open(img_path)

				if scale_factor < 1.0:
					original_width, original_height = img.size
					new_width = max(1, int(original_width * scale_factor))
					new_height = max(1, int(original_height * scale_factor))
					img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
			except Exception as e:
				continue

			img_width, img_height = img.size

			# Find any available position
			position = find_placement_position_fast(
				img_width, img_height, occupancy_grid, occupancy_cell_size,
				canvas_width, canvas_height
			)

			if position is None:
				# No space found - image will go to phase 3
				continue

			paste_x, paste_y = position

			# Paste the image
			if img.mode == "RGBA":
				canvas.paste(img, (paste_x, paste_y), img)
			else:
				img_rgba = img.convert("RGBA")
				canvas.paste(img_rgba, (paste_x, paste_y), img_rgba)

			# Mark occupied
			mark_occupied(occupancy_grid, paste_x, paste_y, img_width, img_height, occupancy_cell_size)

			placed_at_least_once.add(filename)
			placed_images.append(filename)

		print(f"Phase 2 complete: {len(placed_images)} total images placed")
		print(f"  Unique images used: {len(placed_at_least_once)}")
	else:
		print("\nPhase 2: Skipped (allow_duplicates=True, all cells filled in Phase 1)")

	# PHASE 3: Place any remaining images by expanding canvas (only if duplicates disabled)
	still_remaining = [fn for fn in image_filenames if fn not in placed_at_least_once]

	if still_remaining and not allow_duplicates:
		print(f"\nPhase 3: Expanding canvas for {len(still_remaining)} remaining images...")

		# Calculate area needed for remaining images
		remaining_area = sum(images_info[fn]["area"] * (scale_factor ** 2) for fn in still_remaining)

		# Expand canvas in a balanced way (expand both width and height proportionally)
		# This keeps images closer to the main image instead of far to the right
		expansion_area = remaining_area * 1.3  # 30% buffer

		# Calculate expansion maintaining aspect ratio
		expansion_height = int(np.sqrt(expansion_area / aspect_ratio))
		expansion_width = int(expansion_height * aspect_ratio)

		# Distribute expansion: 70% to width, 30% to height for a more compact look
		expansion_width = int(expansion_width * 0.5)  # Reduce horizontal expansion
		expansion_height = int(expansion_height * 0.5)  # Reduce vertical expansion

		new_canvas_width = canvas_width + expansion_width
		new_canvas_height = canvas_height + expansion_height

		# Create new larger canvas
		new_canvas = Image.new("RGBA", (new_canvas_width, new_canvas_height), (0, 0, 0, 0))
		new_canvas.paste(canvas, (0, 0))
		canvas = new_canvas

		# Expand occupancy grid in both dimensions
		new_occupancy_cols = (new_canvas_width + occupancy_cell_size - 1) // occupancy_cell_size
		new_occupancy_rows = (new_canvas_height + occupancy_cell_size - 1) // occupancy_cell_size
		new_occupancy_grid = np.zeros((new_occupancy_rows, new_occupancy_cols), dtype=np.uint8)
		new_occupancy_grid[:occupancy_rows, :occupancy_cols] = occupancy_grid
		occupancy_grid = new_occupancy_grid
		occupancy_cols = new_occupancy_cols
		occupancy_rows = new_occupancy_rows
		canvas_width = new_canvas_width
		canvas_height = new_canvas_height

		print(f"Canvas expanded to: {canvas_width}x{canvas_height}")
		print(f"  Added {expansion_width}px width and {expansion_height}px height")

		# Now place remaining images in the new space
		for filename in tqdm(still_remaining, desc="Placing in expanded area"):
			img_path = os.path.join(IMAGE_FOLDER, filename)
			try:
				img = Image.open(img_path)

				if scale_factor < 1.0:
					original_width, original_height = img.size
					new_width = max(1, int(original_width * scale_factor))
					new_height = max(1, int(original_height * scale_factor))
					img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
			except Exception as e:
				continue

			img_width, img_height = img.size

			# Find position in expanded area
			position = find_placement_position_fast(
				img_width, img_height, occupancy_grid, occupancy_cell_size,
				canvas_width, canvas_height
			)

			if position is None:
				# Still no space? Try reducing image size
				attempt_scale = 0.8
				while attempt_scale > 0.2 and position is None:
					scaled_width = max(1, int(img_width * attempt_scale))
					scaled_height = max(1, int(img_height * attempt_scale))
					temp_img = img.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)

					position = find_placement_position_fast(
						scaled_width, scaled_height, occupancy_grid, occupancy_cell_size,
						canvas_width, canvas_height
					)

					if position:
						img = temp_img
						img_width, img_height = scaled_width, scaled_height
						break

					attempt_scale -= 0.1

				if position is None:
					print(f"\nWarning: Could not place {filename} even after scaling")
					continue

			paste_x, paste_y = position

			# Paste the image
			if img.mode == "RGBA":
				canvas.paste(img, (paste_x, paste_y), img)
			else:
				img_rgba = img.convert("RGBA")
				canvas.paste(img_rgba, (paste_x, paste_y), img_rgba)

			# Mark occupied
			mark_occupied(occupancy_grid, paste_x, paste_y, img_width, img_height, occupancy_cell_size)

			placed_at_least_once.add(filename)
			placed_images.append(filename)

		print(f"Phase 3 complete: {len(placed_images)} total images placed")
		print(f"  Unique images used: {len(placed_at_least_once)}")

	print(f"\n=== FINAL SUMMARY ===")
	print(f"Total images placed: {len(placed_images)}")
	print(f"Unique images used: {len(placed_at_least_once)} out of {num_images} available")

	if allow_duplicates:
		print(f"Duplicates allowed: Yes (images reused to fill grid)")
		duplicate_count = len(placed_images) - len(placed_at_least_once)
		print(f"  Images reused: {duplicate_count} times")

	unique_usage_percentage = (len(placed_at_least_once) / num_images * 100) if num_images > 0 else 0
	print(f"Unique image usage: {unique_usage_percentage:.1f}%")

	if len(placed_at_least_once) < num_images:
		unused_count = num_images - len(placed_at_least_once)
		print(f"\nWarning: {unused_count} unique images could not be placed")
		print("This usually indicates corrupted image files or extreme size constraints")

	# Crop canvas to actual content (optional - remove whitespace)
	# For now, we'll keep the full canvas to show all placed images

	# Save the result
	print(f"\nSaving mosaic to {output_file}...")
	canvas.save(output_file)
	print("[OK] Mosaic saved successfully!")

	return canvas
