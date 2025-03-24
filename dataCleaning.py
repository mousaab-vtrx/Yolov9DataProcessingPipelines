import os
import shutil
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )

def copy_file(src, dst_dir):
    """Copy a file to the given directory and log the operation."""
    try:
        shutil.copy2(src, dst_dir)
        logging.info(f"Copied {src} to {dst_dir}")
    except Exception as e:
        logging.error(f"Error copying {src} to {dst_dir}: {e}")

def extractCleanData(images_dir, masks_dir):
    try:
        # Setup output directories
        base_output = os.path.join(os.getcwd(), "filteredData")
        images_output = os.path.join(base_output, "images")
        masks_output = os.path.join(base_output, "masks")
        os.makedirs(images_output, exist_ok=True)
        os.makedirs(masks_output, exist_ok=True)
        logging.info("Output directories created successfully.")
    except Exception as e:
        logging.error(f"Error creating output directories: {e}")
        return

    try:
        # Build dictionaries for images and masks filtered by naming patterns.
        images = {image[:6]: os.path.join(images_dir, image)
                  for image in os.listdir(images_dir) if image.endswith("_0.jpg")}
        masks = {mask[:6]: os.path.join(masks_dir, mask)
                 for mask in os.listdir(masks_dir) if mask.endswith("_4.png")}
        logging.info(f"Found {len(images)} images and {len(masks)} masks based on naming patterns.")
    except Exception as e:
        logging.error(f"Error reading directories: {e}")
        return

    # Find common keys between images and masks (i.e., matching pairs)
    common_keys = images.keys() & masks.keys()
    logging.info(f"Found {len(common_keys)} matching pairs.")

    # Use a thread pool to copy files concurrently, improving performance.
    tasks = []
    with ThreadPoolExecutor() as executor:
        for key in common_keys:
            # Submit copy tasks for both image and mask files
            tasks.append(executor.submit(copy_file, images[key], images_output))
            tasks.append(executor.submit(copy_file, masks[key], masks_output))
        # Wait for all tasks to complete and catch any exceptions.
        for future in as_completed(tasks):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error in file copying task: {e}")
    
    logging.info("All files copied successfully.")

if __name__ == "__main__":
    setup_logging()

    images_dir = "/content/drive/MyDrive/Colab Notebooks/upper_body/images"
    masks_dir = "/content/drive/MyDrive/Colab Notebooks/upper_body/masks"
    
    # Verify the existence of the source directories.
    if not (os.path.exists(images_dir) and os.path.exists(masks_dir)):
        logging.error("One or both directories do not exist. Please recheck if your directories' paths are correct.")
    else:
        logging.info("Starting data extraction...")
        extractCleanData(images_dir, masks_dir)
        logging.info("The extraction is finished.")
