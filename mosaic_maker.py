from scripts.analyze_images_in_folder import analyze_images_in_folder
from scripts.assemble_mosaic import assemble_mosaic

TARGET_IMAGE = "target.jpg"
IMAGE_FOLDER = "screenshots"

analyze_images_in_folder(IMAGE_FOLDER)
assemble_mosaic(TARGET_IMAGE)
