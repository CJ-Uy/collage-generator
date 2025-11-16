"""
Main mosaic assembly module.

This module coordinates the entire mosaic creation process:
1. Load and filter images
2. Analyze target image for color grid
3. Match images to grid cells by color
4. Place all images ensuring every unique image appears at least once
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# Import our modular utilities
from scripts.image_loader import load_images_info, load_and_scale_image
from scripts.image_filters import filter_images_by_date
from scripts.color_matching import build_color_tree, find_best_match
from scripts.canvas_utils import create_occupancy_grid, mark_occupied, expand_canvas
from scripts.placement import (
	get_spiral_order_cells,
	find_placement_position_fast,
	find_position_near_target
)
from scripts.target_analyzer import analyze_target_image, calculate_optimal_cell_size


def calculate_canvas_dimensions(images_info, aspect_ratio, scale_factor, tight_packing, max_canvas_size):
	"""Calculate optimal canvas dimensions to fit all images."""
	total_area = sum(info["area"] for info in images_info.values())
	print(f"  Total image area: {total_area:,} px²")

	# Apply scaling to total area
	scaled_total_area = int(total_area * (scale_factor**2))

	# Add buffer based on packing strategy
	if tight_packing:
		target_area = int(scaled_total_area * 1.5)  # 50% buffer for tight packing
	else:
		target_area = int(scaled_total_area * 3.0)  # 200% buffer for random packing

	print(f"  Scaled image area: {scaled_total_area:,} px²")
	print(f"  Target canvas area (with buffer): {target_area:,} px²")

	# Calculate dimensions maintaining aspect ratio
	canvas_height = int(np.sqrt(target_area / aspect_ratio))
	canvas_width = int(canvas_height * aspect_ratio)

	# Ensure we don't exceed max_canvas_size
	if max_canvas_size and max(canvas_width, canvas_height) > max_canvas_size:
		adjustment = max_canvas_size / max(canvas_width, canvas_height)
		canvas_width = int(canvas_width * adjustment)
		canvas_height = int(canvas_height * adjustment)
		print("  Applied adjustment to meet size limit")

	return canvas_width, canvas_height


def place_image_on_canvas(canvas, img, position):
	"""Paste image onto canvas handling transparency properly."""
	paste_x, paste_y = position

	if img.mode == "RGBA":
		canvas.paste(img, (paste_x, paste_y), img)
	else:
		img_rgba = img.convert("RGBA")
		canvas.paste(img_rgba, (paste_x, paste_y), img_rgba)


def phase1_color_matched_placement(
	spiral_cells, grid, tree, image_filenames, images_info,
	IMAGE_FOLDER, scale_factor, cell_size, occupancy_grid,
	occupancy_cell_size, canvas, canvas_width, canvas_height,
	num_images, allow_duplicates
):
	"""
	Phase 1: Place images with color matching in spiral order from center.

	Returns:
	    Tuple of (placed_at_least_once set, placed_images list)
	"""
	print("\nPhase 1: Placing images with color matching...")
	placed_at_least_once = set()
	placed_images = []
	used_images = set()

	grid_rows = max(r for r, c in grid.keys()) + 1 if grid else 0
	grid_cols = max(c for r, c in grid.keys()) + 1 if grid else 0

	for row, col in tqdm(spiral_cells, desc="Color matching"):
		# Stop only if duplicates disabled and all images used
		if not allow_duplicates and len(used_images) >= num_images:
			break

		target_color = grid.get((row, col))
		if target_color is None:
			continue

		# Find best matching image
		if allow_duplicates:
			if len(placed_at_least_once) < num_images:
				# Prioritize unused images
				best_match = find_best_match(target_color, tree, image_filenames, placed_at_least_once, k=50)
			else:
				# All images placed once, allow any best match
				best_match = find_best_match(target_color, tree, image_filenames, set(), k=50)
		else:
			# No duplicates - only use each image once
			best_match = find_best_match(target_color, tree, image_filenames, used_images, k=50)

		if best_match is None:
			continue

		# Load and scale image
		img_path = os.path.join(IMAGE_FOLDER, best_match)
		img = load_and_scale_image(img_path, scale_factor)
		if img is None:
			continue

		img_width, img_height = img.size

		# Find position near target cell
		target_x = col * cell_size
		target_y = row * cell_size

		position = find_position_near_target(
			target_x, target_y, img_width, img_height,
			occupancy_grid, occupancy_cell_size,
			canvas_width, canvas_height, cell_size,
			max(grid_rows, grid_cols)
		)

		if position is None:
			continue

		# Place image
		place_image_on_canvas(canvas, img, position)
		mark_occupied(occupancy_grid, position[0], position[1], img_width, img_height, occupancy_cell_size)

		# Track usage
		if not allow_duplicates:
			used_images.add(best_match)
		placed_at_least_once.add(best_match)
		placed_images.append(best_match)

	print(f"Phase 1 complete: {len(placed_images)} images placed")
	print(f"  Unique images used: {len(placed_at_least_once)}")

	return placed_at_least_once, placed_images


def phase2_fill_remaining_gaps(
	image_filenames, placed_at_least_once, images_info,
	IMAGE_FOLDER, scale_factor, occupancy_grid, occupancy_cell_size,
	canvas, canvas_width, canvas_height, placed_images
):
	"""
	Phase 2: Place remaining images in any available space.

	Returns:
	    Updated placed_at_least_once set
	"""
	remaining_images = [fn for fn in image_filenames if fn not in placed_at_least_once]

	if not remaining_images:
		print("\nPhase 2: Skipped (all images already placed in Phase 1)")
		return placed_at_least_once

	print(f"\nPhase 2: Placing {len(remaining_images)} remaining images in available space...")

	# Sort by size (largest first for better packing)
	remaining_sorted = sorted(
		remaining_images,
		key=lambda fn: images_info[fn]["area"],
		reverse=True
	)

	for filename in tqdm(remaining_sorted, desc="Filling gaps"):
		img_path = os.path.join(IMAGE_FOLDER, filename)
		img = load_and_scale_image(img_path, scale_factor)
		if img is None:
			continue

		img_width, img_height = img.size

		position = find_placement_position_fast(
			img_width, img_height, occupancy_grid, occupancy_cell_size,
			canvas_width, canvas_height
		)

		if position is None:
			continue

		place_image_on_canvas(canvas, img, position)
		mark_occupied(occupancy_grid, position[0], position[1], img_width, img_height, occupancy_cell_size)

		placed_at_least_once.add(filename)
		placed_images.append(filename)

	print(f"Phase 2 complete: {len(placed_images)} total images placed")
	print(f"  Unique images used: {len(placed_at_least_once)}")

	return placed_at_least_once


def phase3_expand_and_place(
	image_filenames, placed_at_least_once, images_info,
	IMAGE_FOLDER, scale_factor, occupancy_grid, occupancy_cell_size,
	canvas, canvas_width, canvas_height, placed_images,
	aspect_ratio, occupancy_rows, occupancy_cols
):
	"""
	Phase 3: Expand canvas and place any remaining images.

	Returns:
	    Tuple of (canvas, occupancy_grid, canvas_width, canvas_height, occupancy_rows, occupancy_cols, placed_at_least_once)
	"""
	still_remaining = [fn for fn in image_filenames if fn not in placed_at_least_once]

	if not still_remaining:
		return canvas, occupancy_grid, canvas_width, canvas_height, occupancy_rows, occupancy_cols, placed_at_least_once

	print(f"\nPhase 3: Expanding canvas for {len(still_remaining)} remaining images...")

	# Calculate expansion needed
	remaining_area = sum(images_info[fn]["area"] * (scale_factor ** 2) for fn in still_remaining)
	expansion_area = remaining_area * 1.3  # 30% buffer

	# Calculate balanced expansion
	expansion_height = int(np.sqrt(expansion_area / aspect_ratio))
	expansion_width = int(expansion_height * aspect_ratio)

	# Reduce expansion to keep images closer (50% of calculated)
	expansion_width = int(expansion_width * 0.5)
	expansion_height = int(expansion_height * 0.5)

	# Expand canvas
	canvas, occupancy_grid, canvas_width, canvas_height, occupancy_rows, occupancy_cols = expand_canvas(
		canvas, occupancy_grid, occupancy_cell_size, expansion_width, expansion_height
	)

	print(f"Canvas expanded to: {canvas_width}x{canvas_height}")
	print(f"  Added {expansion_width}px width and {expansion_height}px height")

	# Place remaining images
	for filename in tqdm(still_remaining, desc="Placing in expanded area"):
		img_path = os.path.join(IMAGE_FOLDER, filename)
		img = load_and_scale_image(img_path, scale_factor)
		if img is None:
			continue

		img_width, img_height = img.size

		position = find_placement_position_fast(
			img_width, img_height, occupancy_grid, occupancy_cell_size,
			canvas_width, canvas_height
		)

		# Try scaling down if no position found
		if position is None:
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

		place_image_on_canvas(canvas, img, position)
		mark_occupied(occupancy_grid, position[0], position[1], img_width, img_height, occupancy_cell_size)

		placed_at_least_once.add(filename)
		placed_images.append(filename)

	print(f"Phase 3 complete: {len(placed_images)} total images placed")
	print(f"  Unique images used: {len(placed_at_least_once)}")

	return canvas, occupancy_grid, canvas_width, canvas_height, occupancy_rows, occupancy_cols, placed_at_least_once


def print_final_summary(placed_images, placed_at_least_once, num_images, allow_duplicates):
	"""Print final statistics about the mosaic."""
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
	overlay_opacity=0.0,
):
	"""
	Assemble a mosaic from differently-sized images to recreate a target image.

	Args:
	    TARGET_FILENAME: Path to target image
	    IMAGE_FOLDER: Folder containing source images
	    cell_size: Size of grid cells for target analysis. If None, auto-calculates
	    output_file: Output filename for the mosaic
	    start_date: Start date for filtering (YYYY-MM-DD) or None
	    end_date: End date for filtering (YYYY-MM-DD) or None
	    use_all_images: If True, sizes canvas to fit all images
	    scale_factor: Scale images by this factor. If None, auto-calculates
	    max_canvas_size: Maximum dimension in pixels (default 20000)
	    allow_overlaps: Allow images to overlap (default False)
	    tight_packing: Use sequential placement for tight packing (default True)
	    allow_duplicates: Allow image reuse while ensuring all appear once (default False)
	    overlay_opacity: Opacity of target image overlay (0.0-1.0, 0=none, 1=full). Default 0.0

	Returns:
	    PIL Image object of the completed mosaic
	"""
	print("\n=== MOSAIC ASSEMBLY ===")
	print(f"Target image: {TARGET_FILENAME}")
	print(f"Max canvas size: {max_canvas_size:,} px per dimension" if max_canvas_size else "Max canvas size: Unlimited")

	# Load and filter images
	print("\nLoading image information...")
	images_info = load_images_info()
	print(f"Loaded info for {len(images_info)} images")

	if start_date or end_date:
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
		return None

	# Build color matching tree
	image_filenames = list(images_info.keys())
	tree = build_color_tree(images_info, image_filenames)

	# Analyze target image
	target_img = Image.open(TARGET_FILENAME)
	target_width_orig, target_height_orig = target_img.size
	aspect_ratio = target_width_orig / target_height_orig
	print(f"\nTarget image aspect ratio: {aspect_ratio:.2f}")

	# Calculate canvas dimensions
	if use_all_images:
		print("\nCalculating optimal canvas size...")

		# Auto-calculate scale_factor if needed
		if scale_factor is None:
			total_area = sum(info["area"] for info in images_info.values())
			if tight_packing:
				ideal_area = int(total_area * 1.5)
			else:
				ideal_area = int(total_area * 3.0)

			ideal_height = int(np.sqrt(ideal_area / aspect_ratio))
			ideal_width = int(ideal_height * aspect_ratio)
			max_dim = max(ideal_width, ideal_height)

			if max_dim > max_canvas_size:
				scale_factor = max_canvas_size / max_dim
				print(f"  Auto-scaling to fit within {max_canvas_size}px limit")
				print(f"  Calculated scale factor: {scale_factor * 100:.1f}%")
			else:
				scale_factor = 1.0
				print("  Canvas fits within limit, no scaling needed")
		else:
			print(f"  Using user-specified scale factor: {scale_factor * 100:.0f}%")

		canvas_width, canvas_height = calculate_canvas_dimensions(
			images_info, aspect_ratio, scale_factor, tight_packing, max_canvas_size
		)
		print(f"Final canvas size: {canvas_width}x{canvas_height} ({canvas_width * canvas_height:,} px²)")
	else:
		canvas_width = target_width_orig * 2
		canvas_height = target_height_orig * 2
		if scale_factor is None:
			scale_factor = 1.0

	# Calculate cell size
	if cell_size is None:
		if use_all_images:
			if tight_packing:
				total_cells_needed = int(num_images * 1.2)
			else:
				total_cells_needed = int(num_images * 2.5)
			cell_area = (canvas_width * canvas_height) / total_cells_needed
			cell_size = max(10, int(np.sqrt(cell_area)))
			print(f"\nCalculated cell size: {cell_size}px for {num_images} images")
		else:
			cell_size = calculate_optimal_cell_size(images_info)
			print(f"\nCalculated cell size: {cell_size}px")

	# Analyze target image grid
	print("\nAnalyzing target image grid...")
	grid_rows = canvas_height // cell_size
	grid_cols = canvas_width // cell_size

	target_img_resized = target_img.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
	np_img = np.array(target_img_resized.convert("RGB"))

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

	# Create canvas and occupancy grid
	canvas = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))

	if tight_packing:
		occupancy_cell_size = max(5, cell_size // 12)
	else:
		occupancy_cell_size = max(15, cell_size // 8)

	occupancy_grid, occupancy_rows, occupancy_cols = create_occupancy_grid(
		canvas_width, canvas_height, occupancy_cell_size
	)
	print(f"Occupancy grid: {occupancy_rows}x{occupancy_cols} cells of {occupancy_cell_size}px")

	# Get spiral order for placement
	spiral_cells = list(get_spiral_order_cells(grid_rows, grid_cols))
	placed_images = []

	# Execute three-phase placement
	placed_at_least_once, placed_images = phase1_color_matched_placement(
		spiral_cells, grid, tree, image_filenames, images_info,
		IMAGE_FOLDER, scale_factor, cell_size, occupancy_grid,
		occupancy_cell_size, canvas, canvas_width, canvas_height,
		num_images, allow_duplicates
	)

	placed_at_least_once = phase2_fill_remaining_gaps(
		image_filenames, placed_at_least_once, images_info,
		IMAGE_FOLDER, scale_factor, occupancy_grid, occupancy_cell_size,
		canvas, canvas_width, canvas_height, placed_images
	)

	canvas, occupancy_grid, canvas_width, canvas_height, occupancy_rows, occupancy_cols, placed_at_least_once = phase3_expand_and_place(
		image_filenames, placed_at_least_once, images_info,
		IMAGE_FOLDER, scale_factor, occupancy_grid, occupancy_cell_size,
		canvas, canvas_width, canvas_height, placed_images,
		aspect_ratio, occupancy_rows, occupancy_cols
	)

	# Print summary
	print_final_summary(placed_images, placed_at_least_once, num_images, allow_duplicates)

	# Apply target image overlay if requested
	if overlay_opacity > 0:
		print(f"\nApplying target image overlay (opacity: {overlay_opacity * 100:.0f}%)...")

		# Resize target image to match final canvas size
		target_overlay = target_img.resize((canvas.size[0], canvas.size[1]), Image.Resampling.LANCZOS)

		# Convert to RGBA if needed
		if target_overlay.mode != "RGBA":
			target_overlay = target_overlay.convert("RGBA")

		# Adjust opacity
		overlay_with_alpha = target_overlay.copy()
		alpha = overlay_with_alpha.split()[3]
		alpha = alpha.point(lambda p: int(p * overlay_opacity))
		overlay_with_alpha.putalpha(alpha)

		# Create final composite
		final_canvas = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
		final_canvas.paste(canvas, (0, 0))
		final_canvas = Image.alpha_composite(final_canvas, overlay_with_alpha)
		canvas = final_canvas

		print("[OK] Overlay applied successfully!")

	# Save result
	print(f"\nSaving mosaic to {output_file}...")
	canvas.save(output_file)
	print("[OK] Mosaic saved successfully!")

	return canvas
