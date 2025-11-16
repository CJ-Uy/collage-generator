"""Canvas and occupancy grid management utilities."""

import numpy as np
from PIL import Image


def create_occupancy_grid(canvas_width, canvas_height, cell_size):
	"""
	Create an occupancy grid for tracking placed images.

	Args:
	    canvas_width: Width of canvas in pixels
	    canvas_height: Height of canvas in pixels
	    cell_size: Size of each grid cell in pixels

	Returns:
	    Tuple of (occupancy_grid, rows, cols)
	"""
	rows = (canvas_height + cell_size - 1) // cell_size
	cols = (canvas_width + cell_size - 1) // cell_size
	grid = np.zeros((rows, cols), dtype=np.uint8)
	return grid, rows, cols


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


def expand_canvas(canvas, occupancy_grid, occupancy_cell_size, expansion_width, expansion_height):
	"""
	Expand canvas and occupancy grid in both dimensions.

	Args:
	    canvas: Current PIL Image canvas
	    occupancy_grid: Current occupancy grid
	    occupancy_cell_size: Size of occupancy grid cells
	    expansion_width: Pixels to add to width
	    expansion_height: Pixels to add to height

	Returns:
	    Tuple of (new_canvas, new_occupancy_grid, new_width, new_height, new_rows, new_cols)
	"""
	current_width, current_height = canvas.size
	occupancy_rows, occupancy_cols = occupancy_grid.shape

	# Calculate new dimensions
	new_canvas_width = current_width + expansion_width
	new_canvas_height = current_height + expansion_height

	# Create new larger canvas
	new_canvas = Image.new("RGBA", (new_canvas_width, new_canvas_height), (0, 0, 0, 0))
	new_canvas.paste(canvas, (0, 0))

	# Expand occupancy grid in both dimensions
	new_occupancy_cols = (new_canvas_width + occupancy_cell_size - 1) // occupancy_cell_size
	new_occupancy_rows = (new_canvas_height + occupancy_cell_size - 1) // occupancy_cell_size
	new_occupancy_grid = np.zeros((new_occupancy_rows, new_occupancy_cols), dtype=np.uint8)
	new_occupancy_grid[:occupancy_rows, :occupancy_cols] = occupancy_grid

	return (
		new_canvas,
		new_occupancy_grid,
		new_canvas_width,
		new_canvas_height,
		new_occupancy_rows,
		new_occupancy_cols
	)
