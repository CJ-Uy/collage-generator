"""Color matching and k-d tree utilities."""

from scipy.spatial import cKDTree


def build_color_tree(images_info, image_filenames):
	"""
	Build a k-d tree for fast color matching.

	Args:
	    images_info: Dict of image info from JSON
	    image_filenames: List of image filenames

	Returns:
	    cKDTree object for color queries
	"""
	image_colors = [images_info[fn]["average_rgb"] for fn in image_filenames]
	return cKDTree(image_colors)


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
