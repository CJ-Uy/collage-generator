import os
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS


def get_file_date(file_path):
	"""
	Extract the date a photo was taken, with multiple fallback strategies.

	Priority order:
	1. EXIF DateTimeOriginal (when photo was taken)
	2. EXIF DateTime (when photo was modified)
	3. EXIF DateTimeDigitized (when photo was digitized)
	4. File modification time (last resort)

	Args:
	    file_path: Path to image file

	Returns:
	    Date string in format "YYYY-MM-DD HH:MM:SS" or None
	"""
	# Try EXIF data first (best for Apple photos and most digital cameras)
	try:
		with Image.open(file_path) as img:
			exif_data = img._getexif()

			if exif_data:
				# Create a mapping of tag names to values
				exif = {TAGS.get(tag, tag): value for tag, value in exif_data.items()}

				# Try different EXIF date fields in order of preference
				date_fields = [
					'DateTimeOriginal',  # When photo was taken (best)
					'DateTime',          # General datetime
					'DateTimeDigitized', # When photo was scanned/digitized
				]

				for field in date_fields:
					if field in exif:
						date_str = exif[field]
						# EXIF format: "YYYY:MM:DD HH:MM:SS"
						# Convert to our format: "YYYY-MM-DD HH:MM:SS"
						try:
							date_obj = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
							return date_obj.strftime("%Y-%m-%d %H:%M:%S")
						except ValueError:
							continue
	except Exception:
		pass  # Fall through to file modification time

	# Fallback: Use file modification time
	try:
		mod_timestamp = os.path.getmtime(file_path)
		return datetime.fromtimestamp(mod_timestamp).strftime("%Y-%m-%d %H:%M:%S")
	except OSError:
		return None
