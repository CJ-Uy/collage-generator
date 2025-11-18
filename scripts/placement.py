"""Image placement algorithms."""

import numpy as np


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
			if (
				0 <= row < grid_rows
				and 0 <= col < grid_cols
				and (row, col) not in visited
			):
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


def find_placement_position_fast(
	img_width, img_height, occupancy_grid, grid_size, canvas_width, canvas_height
):
	"""
	Find a valid position to place an image using a fast occupancy grid.

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
			region = occupancy_grid[
				row : row + cells_needed_h, col : col + cells_needed_w
			]
			if np.any(region):
				continue

			# Calculate pixel position - align to grid for tight packing
			px = col * grid_size
			py = row * grid_size

			# Ensure we stay in bounds
			px = max(0, min(px, canvas_width - img_width))
			py = max(0, min(py, canvas_height - img_height))

			# Check canvas bounds
			if (
				px + img_width > canvas_width
				or py + img_height > canvas_height
				or px < 0
				or py < 0
			):
				continue

			return (px, py)

	return None


def find_position_near_target(
	target_x,
	target_y,
	img_width,
	img_height,
	occupancy_grid,
	occupancy_cell_size,
	canvas_width,
	canvas_height,
	cell_size,
	max_radius,
):
	"""
	Find placement position near a target location.

	Args:
	    target_x, target_y: Target position
	    img_width, img_height: Image dimensions
	    occupancy_grid: Occupancy grid
	    occupancy_cell_size: Size of occupancy cells
	    canvas_width, canvas_height: Canvas dimensions
	    cell_size: Grid cell size for search
	    max_radius: Maximum search radius

	Returns:
	    (x, y) position or None
	"""
	occupancy_rows, occupancy_cols = occupancy_grid.shape

	# Search in expanding radius around target position
	for search_radius in range(0, max_radius):
		# Try positions in a square around target
		for dy in range(-search_radius, search_radius + 1):
			for dx in range(-search_radius, search_radius + 1):
				# Only check perimeter of square
				if (
					search_radius > 0
					and abs(dx) != search_radius
					and abs(dy) != search_radius
				):
					continue

				check_x = max(
					0, min(target_x + dx * cell_size, canvas_width - img_width)
				)
				check_y = max(
					0, min(target_y + dy * cell_size, canvas_height - img_height)
				)

				# Check if position is valid in occupancy grid
				start_col_occ = check_x // occupancy_cell_size
				start_row_occ = check_y // occupancy_cell_size
				cells_needed_h = (
					img_height + occupancy_cell_size - 1
				) // occupancy_cell_size
				cells_needed_w = (
					img_width + occupancy_cell_size - 1
				) // occupancy_cell_size

				if (
					start_row_occ + cells_needed_h > occupancy_rows
					or start_col_occ + cells_needed_w > occupancy_cols
				):
					continue

				region = occupancy_grid[
					start_row_occ : start_row_occ + cells_needed_h,
					start_col_occ : start_col_occ + cells_needed_w,
				]
				if not np.any(region):
					return (check_x, check_y)

	return None
