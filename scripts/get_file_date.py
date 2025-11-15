import os
from datetime import datetime

def get_file_date(file_path):
	try:
		# Get modification time as a Unix timestamp
		mod_timestamp = os.path.getmtime(file_path)
		# Convert timestamp to a readable string format
		return datetime.fromtimestamp(mod_timestamp).strftime("%Y-%m-%d %H:%M:%S")
	except OSError:
		return None
