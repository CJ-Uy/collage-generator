"""Batch processing utilities for multiple target images."""

import os
import re
from tqdm import tqdm
from scripts.assemble_mosaic import assemble_mosaic


def parse_date_from_filename(filename):
	"""
	Extract start and end dates from filename.

	Expected format: "YYYY-MM-DD to YYYY-MM-DD.jpg" or "YYYY-MM-DD to YYYY-MM-DD.png"

	Returns:
	    Tuple of (start_date, end_date, base_name) or (None, None, None)
	"""
	# Remove file extension
	name_without_ext = os.path.splitext(filename)[0]

	# Pattern: "YYYY-MM-DD to YYYY-MM-DD"
	pattern = r'(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})'
	match = re.match(pattern, name_without_ext)

	if match:
		start_date = match.group(1)
		end_date = match.group(2)
		return start_date, end_date, name_without_ext
	else:
		return None, None, None


def process_targets_folder(
	targets_folder="targets",
	screenshots_folder="screenshots",
	outputs_folder="outputs",
	overlay_opacity=0.3,
	verbose=False
):
	"""
	Process all target images in the targets folder.

	Args:
	    targets_folder: Folder containing target images
	    screenshots_folder: Folder containing source screenshots
	    outputs_folder: Folder for output mosaics
	    overlay_opacity: Opacity of target image overlay (0.0-1.0)
	    verbose: If True, show detailed output. If False, show one progress bar per image
	"""
	# Check if targets folder exists
	if not os.path.exists(targets_folder):
		print(f"ERROR: '{targets_folder}' folder not found!")
		print(f"Please create a '{targets_folder}' folder and add target images.")
		print(f"\nTarget image filenames should be in format:")
		print(f"  'YYYY-MM-DD to YYYY-MM-DD.jpg'")
		print(f"\nExample:")
		print(f"  '2023-01-01 to 2023-12-31.jpg'")
		return

	# Create outputs folder if it doesn't exist
	if not os.path.exists(outputs_folder):
		os.makedirs(outputs_folder)
		if verbose:
			print(f"Created '{outputs_folder}' folder")

	# Find all image files in targets folder
	supported_formats = ('.jpg', '.jpeg', '.png')
	target_files = [
		f for f in os.listdir(targets_folder)
		if f.lower().endswith(supported_formats)
	]

	if not target_files:
		print(f"No target images found in '{targets_folder}' folder!")
		return

	if verbose:
		print("=" * 70)
		print(f"Found {len(target_files)} target image(s)")
		print("=" * 70)
	else:
		print(f"\nProcessing {len(target_files)} target image(s)...")

	# Process each target image
	processed = 0
	skipped = 0

	# Use tqdm to show overall progress when not verbose
	target_files_iter = tqdm(target_files, desc="Overall progress", disable=verbose) if not verbose else target_files

	for target_file in target_files_iter:
		if verbose:
			print("\n" + "=" * 70)
			print(f"Processing: {target_file}")
			print("=" * 70)
		else:
			# Update the progress bar description with current file
			if hasattr(target_files_iter, 'set_description'):
				target_files_iter.set_description(f"Processing: {target_file[:40]}")

		# Parse dates from filename
		start_date, end_date, base_name = parse_date_from_filename(target_file)

		if start_date is None:
			if verbose:
				print(f"⚠ SKIPPED: Filename doesn't match expected format")
				print(f"  Expected: 'YYYY-MM-DD to YYYY-MM-DD.jpg'")
				print(f"  Got: '{target_file}'")
			else:
				tqdm.write(f"⚠ SKIPPED: {target_file} (invalid format)")
			skipped += 1
			continue

		# Build paths
		target_path = os.path.join(targets_folder, target_file)
		output_filename = f"{base_name} Mosaic.png"
		output_path = os.path.join(outputs_folder, output_filename)

		if verbose:
			print(f"\nDate range: {start_date} to {end_date}")
			print(f"Output: {output_path}")

		# Create the mosaic
		try:
			assemble_mosaic(
				target_path,
				IMAGE_FOLDER=screenshots_folder,
				output_file=output_path,
				start_date=start_date,
				end_date=end_date,
				overlay_opacity=overlay_opacity,
				# Auto-optimized settings
				scale_factor=None,
				max_canvas_size=50000,
				use_all_images=True,
				tight_packing=True,
				allow_duplicates=False,
				allow_overlaps=False,
				verbose=verbose,
				show_progress=True,  # Always show progress bars
			)

			processed += 1
			if verbose:
				print(f"\n✓ Successfully created: {output_filename}")
			else:
				tqdm.write(f"✓ {output_filename}")

		except Exception as e:
			if verbose:
				print(f"\n✗ ERROR processing {target_file}: {e}")
			else:
				tqdm.write(f"✗ ERROR: {target_file}: {e}")
			skipped += 1

	# Final summary
	print("\n" + "=" * 70)
	print("BATCH PROCESSING COMPLETE")
	print("=" * 70)
	print(f"✓ Successfully processed: {processed}")
	if skipped > 0:
		print(f"⚠ Skipped: {skipped}")
	print(f"\nOutput mosaics saved to: {outputs_folder}/")
	print("=" * 70)
