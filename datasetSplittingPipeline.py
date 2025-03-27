import os
import shutil
import logging
import traceback
import time
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_organization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def validate_and_organize_data(image_dir, label_dir, output_base_dir, val_size=0.10, test_size=0.05, random_state=42):
    """
    Validates, organizes, and renames images and labels for YOLOv9 training.

    Args:
        image_dir: Path to the directory containing images.
        label_dir: Path to the directory containing labels.
        output_base_dir: Base directory for the organized data.
        val_size: Proportion of data to use for validation (default: 0.10).
        test_size: Proportion of data to use for testing (default: 0.05).
        random_state: Random seed for train-test split (default: 42).

    Returns:
        True if validation and organization were successful, False otherwise.
    """
    start_time = time.time()
    logger.info(f"Starting data validation and organization process")
    logger.info(f"Parameters: val_size={val_size}, test_size={test_size}, random_state={random_state}")
    
    try:
        # Validate input directories exist
        if not os.path.exists(image_dir):
            logger.error(f"Image directory does not exist: {image_dir}")
            return False
            
        if not os.path.exists(label_dir):
            logger.error(f"Label directory does not exist: {label_dir}")
            return False
        
        # Create output directories
        images_train_dir = os.path.join(output_base_dir, "images", "train")
        images_val_dir = os.path.join(output_base_dir, "images", "val")
        images_test_dir = os.path.join(output_base_dir, "images", "test")
        labels_train_dir = os.path.join(output_base_dir, "labels", "train")
        labels_val_dir = os.path.join(output_base_dir, "labels", "val")
        labels_test_dir = os.path.join(output_base_dir, "labels", "test")

        logger.debug(f"Creating output directories")
        for dir_path in [images_train_dir, images_val_dir, images_test_dir, 
                         labels_train_dir, labels_val_dir, labels_test_dir]:
            try:
                os.makedirs(dir_path, exist_ok=True)
                logger.debug(f"Created directory: {dir_path}")
            except Exception as e:
                logger.error(f"Failed to create directory {dir_path}: {str(e)}")
                return False

        # Get files from directories
        logger.info("Collecting image and label files")
        try:
            image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            logger.info(f"Found {len(image_files)} image files")
        except Exception as e:
            logger.error(f"Error accessing image directory: {str(e)}")
            return False
            
        try:
            label_files = [f for f in os.listdir(label_dir) if f.lower().endswith('.txt')]
            logger.info(f"Found {len(label_files)} label files")
        except Exception as e:
            logger.error(f"Error accessing label directory: {str(e)}")
            return False

        # Match files based on the first 6 characters of their names
        logger.info("Matching image and label files")
        matched_images = []
        matched_labels = []
        label_dict = {f[:6]: f for f in label_files}
        
        unmatched_images = []
        for img in image_files:
            prefix = img[:6]
            if prefix in label_dict:
                matched_images.append(img)
                matched_labels.append(label_dict[prefix])
            else:
                unmatched_images.append(img)
        
        logger.info(f"Matched {len(matched_images)} image-label pairs")
        if unmatched_images:
            logger.warning(f"Found {len(unmatched_images)} images without matching labels")
            logger.debug(f"Unmatched images: {unmatched_images[:10]}{'...' if len(unmatched_images) > 10 else ''}")

        # Check for mismatches
        if len(matched_images) != len(matched_labels):
            logger.error(f"Number of matched images ({len(matched_images)}) does not match number of matched labels ({len(matched_labels)})")
            return False
            
        if len(matched_images) == 0:
            logger.error("No matching image-label pairs found")
            return False

        # First split data into train and temporary set (val+test combined)
        temp_size = val_size + test_size
        train_images, temp_images, train_labels, temp_labels = train_test_split(
            matched_images, matched_labels, test_size=temp_size, random_state=random_state
        )
        
        # Then split the temporary set into validation and test sets
        # Adjust the test_size to get the right proportion relative to the temp set
        adjusted_test_size = test_size / temp_size
        val_images, test_images, val_labels, test_labels = train_test_split(
            temp_images, temp_labels, test_size=adjusted_test_size, random_state=random_state
        )
        
        logger.info(f"Split result: {len(train_images)} training samples ({len(train_images)/len(matched_images):.1%}), "
                   f"{len(val_images)} validation samples ({len(val_images)/len(matched_images):.1%}), "
                   f"{len(test_images)} test samples ({len(test_images)/len(matched_images):.1%})")

        # Copy and rename files with indexing
        logger.info("Copying and renaming training files")
        if not copy_and_rename_files(train_images, train_labels, image_dir, label_dir, images_train_dir, labels_train_dir):
            return False
            
        logger.info("Copying and renaming validation files")
        if not copy_and_rename_files(val_images, val_labels, image_dir, label_dir, images_val_dir, labels_val_dir):
            return False
            
        logger.info("Copying and renaming test files")
        if not copy_and_rename_files(test_images, test_labels, image_dir, label_dir, images_test_dir, labels_test_dir):
            return False

        duration = time.time() - start_time
        logger.info(f"Data split complete in {duration:.2f} seconds")
        logger.info(f"Train: {len(train_images)} images and {len(train_labels)} labels ({len(train_images)/len(matched_images):.1%})")
        logger.info(f"Val: {len(val_images)} images and {len(val_labels)} labels ({len(val_images)/len(matched_images):.1%})")
        logger.info(f"Test: {len(test_images)} images and {len(test_labels)} labels ({len(test_images)/len(matched_images):.1%})")
        
        print(f"Data split complete.")
        print(f"Train: {len(train_images)} images and {len(train_labels)} labels ({len(train_images)/len(matched_images):.1%})")
        print(f"Val: {len(val_images)} images and {len(val_labels)} labels ({len(val_images)/len(matched_images):.1%})")
        print(f"Test: {len(test_images)} images and {len(test_labels)} labels ({len(test_images)/len(matched_images):.1%})")

        return True
        
    except Exception as e:
        logger.error(f"Unexpected error during data organization: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

def copy_and_rename_files(image_files, label_files, image_src_dir, label_src_dir, image_dest_dir, label_dest_dir):
    """
    Copies and renames files with sequential numbering.
    
    Returns:
        True if successful, False otherwise.
    """
    logger.debug(f"Starting to copy {len(image_files)} files")
    success_count = 0
    error_count = 0
    
    try:
        for index, (image_file, label_file) in enumerate(zip(image_files, label_files), start=1):
            # Create zero-padded filenames
            new_name = f"{index:07d}"
            
            try:
                # Copy and rename image
                image_ext = os.path.splitext(image_file)[1]
                image_src = os.path.join(image_src_dir, image_file)
                image_dest = os.path.join(image_dest_dir, f"{new_name}{image_ext}")
                
                if not os.path.exists(image_src):
                    logger.warning(f"Source image does not exist: {image_src}")
                    error_count += 1
                    continue
                    
                shutil.copy(image_src, image_dest)
                
                # Copy and rename label
                label_ext = os.path.splitext(label_file)[1]
                label_src = os.path.join(label_src_dir, label_file)
                label_dest = os.path.join(label_dest_dir, f"{new_name}{label_ext}")
                
                if not os.path.exists(label_src):
                    logger.warning(f"Source label does not exist: {label_src}")
                    error_count += 1
                    continue
                    
                shutil.copy(label_src, label_dest)
                
                success_count += 1
                
                # Log progress periodically
                if index % 100 == 0 or index == len(image_files):
                    logger.debug(f"Processed {index}/{len(image_files)} files")
                
            except Exception as e:
                logger.error(f"Error copying files for index {index}: {str(e)}")
                error_count += 1
                
        logger.info(f"File copy summary: {success_count} successful, {error_count} failed")
        
        if error_count > 0:
            logger.warning(f"Some files could not be copied ({error_count} failures)")
            
        return success_count > 0  # Return True if at least some files were copied successfully
        
    except Exception as e:
        logger.error(f"Unexpected error during file copying: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

# Main execution block with error handling
if __name__ == "__main__":
    try:
        logger.info("Script execution started")
        
        image_dir = '/content/myData/images'
        label_dir = '/content/labels'
        output_base_dir = '/content/train_data'
        
        logger.info(f"Parameters: image_dir={image_dir}, label_dir={label_dir}, output_base_dir={output_base_dir}")
        
        result = validate_and_organize_data(
            image_dir, 
            label_dir, 
            output_base_dir,
            val_size=0.10,  # 10% validation set
            test_size=0.05  # 5% test set
        )
        
        if result:
            logger.info("Indexing validation successful and data organized")
            print("Indexing validation successful and data organized.")
        else:
            logger.error("Indexing validation failed or data organization incomplete")
            print("Indexing validation failed or data organization incomplete. Check the logs for details.")
            
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        print("\nProcess interrupted by user")
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}")
        logger.critical(traceback.format_exc())
        print(f"An error occurred: {str(e)}")
        print("Check the log file for details.")