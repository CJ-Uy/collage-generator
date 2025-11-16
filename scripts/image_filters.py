"""Image filtering utilities (date, size, etc.)."""

from datetime import datetime


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

	# Check if dates are in wrong order
	if start_date is not None and end_date is not None and start_date > end_date:
		print(f"\nWARNING: START_DATE ({start_date.strftime('%Y-%m-%d')}) is AFTER END_DATE ({end_date.strftime('%Y-%m-%d')})")
		print("Automatically swapping dates to correct the range...")
		start_date, end_date = end_date, start_date

	filtered = {}
	skipped_no_date = 0
	skipped_out_of_range = 0

	for filename, info in images_info.items():
		date_str = info.get("date_taken")
		if not date_str:
			# Skip images without date information
			skipped_no_date += 1
			continue

		try:
			# Parse the date_taken field (format: "YYYY-MM-DD HH:MM:SS")
			img_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
		except (ValueError, TypeError):
			# Skip images with invalid date format
			skipped_no_date += 1
			continue

		# Check if date is within range
		if start_date is not None and img_date < start_date:
			skipped_out_of_range += 1
			continue
		if end_date is not None and img_date > end_date:
			skipped_out_of_range += 1
			continue

		filtered[filename] = info

	# Print helpful statistics
	if skipped_no_date > 0:
		print(f"  Skipped {skipped_no_date} images without date information")
	if skipped_out_of_range > 0:
		print(f"  Skipped {skipped_out_of_range} images outside date range")

	return filtered
