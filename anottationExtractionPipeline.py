import os
import cv2
import numpy as np
import logging
import time
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("segmentation_conversion.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def convert_segmentation_mask(image_path, output_path, color_to_class, min_area=400, epsilon=0.0005):
    """
    Convert a segmentation mask image to YOLOv8 format.
    
    Args:
        image_path: Path to the segmentation mask image
        output_path: Path where the output text file will be saved
        color_to_class: Dictionary mapping color tuples to class IDs
        min_area: Minimum contour area to be considered (default: 400)
        epsilon: Parameter for polygon approximation (default: 0.0005)
    """
    start_time = time.time()
    logger.info(f"Processing image: {image_path}")
    
    try:
        # Read the image with error handling
        mask = cv2.imread(image_path)
        if mask is None:
            logger.error(f"Error: Could not read image {image_path}.")
            return
        
        height, width, _ = mask.shape
        logger.debug(f"Image dimensions: {width}x{height}")
        
        try:
            with open(output_path, 'w') as f:
                for color, class_id in color_to_class.items():
                    logger.debug(f"Processing color {color} for class {class_id}")
                    
                    # Create color bounds - keeping the exact original implementation
                    lower_bound = np.array(color, dtype="uint8")
                    upper_bound = np.array(color, dtype="uint8")
                    
                    # Apply the same exact processing steps as the original
                    color_mask = cv2.inRange(mask, lower_bound, upper_bound)
                    ret, color_mask = cv2.threshold(color_mask, 200, 255, cv2.THRESH_BINARY)
                    color_mask = cv2.erode(color_mask, (7,7), iterations=3)
                    color_mask = cv2.medianBlur(color_mask, 15)
                    
                    # Find contours with error handling
                    try:
                        contours, hierarchy = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                        logger.debug(f"Found {len(contours)} contours for class {class_id}")
                    except Exception as e:
                        logger.error(f"Error finding contours: {str(e)}")
                        continue
                    
                    # Process contours
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area > min_area:
                            try:
                                # Calculate perimeter for approxPolyDP
                                perimeter = cv2.arcLength(cnt, True)
                                if perimeter <= 0:
                                    logger.warning(f"Skipping contour with zero perimeter")
                                    continue
                                    
                                # Apply polygon approximation (keeping original epsilon)
                                smoothed_cnt = cv2.approxPolyDP(cnt, epsilon * perimeter, True)
                                
                                # Generate normalized polygon coordinates
                                try:
                                    polygon = [
                                        f"{x / width:.6f} {y / height:.6f}" for [x, y] in smoothed_cnt[:, 0, :]
                                    ]
                                    f.write(f"{class_id} {' '.join(polygon)}\n")
                                except IndexError as e:
                                    logger.error(f"Error processing polygon points: {str(e)}")
                            except Exception as e:
                                logger.error(f"Error processing contour: {str(e)}")
                    
        except IOError as e:
            logger.error(f"IO error when writing to {output_path}: {str(e)}")
            
        duration = time.time() - start_time
        logger.info(f"Completed processing {image_path} in {duration:.3f}s")
        
    except Exception as e:
        logger.error(f"Unexpected error processing {image_path}: {str(e)}")
        logger.debug(traceback.format_exc())

# Main execution with error handling
if __name__ == "__main__":
    try:
        color_to_class = {
            # (128, 0, 0): 0, #upperbody class
            # (128, 128, 0): 1, #lowerbody class
            (128,128,128): 2  #dresses class 
        }

        input_dir = '/content/masks'
        output_dir = '/content/labels'

        logger.info(f"Starting segmentation mask conversion from {input_dir} to {output_dir}")
        
        # Create output directory
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.debug(f"Created output directory: {output_dir}")
        except Exception as e:
            logger.error(f"Error creating output directory {output_dir}: {str(e)}")
            raise

        # Process each file
        total_files = 0
        processed_files = 0
        start_time = time.time()
        
        try:
            file_list = os.listdir(input_dir)
            total_files = len(file_list)
            logger.info(f"Found {total_files} files to process")
            
            for j in file_list:
                image_path = os.path.join(input_dir, j)
                output_path = os.path.join(output_dir, j.rsplit('.', 1)[0] + '.txt')
                
                # Process the image
                convert_segmentation_mask(image_path, output_path, color_to_class)
                processed_files += 1
                logger.debug(f"Processed {processed_files}/{total_files} files")
                
        except Exception as e:
            logger.error(f"Error during batch processing: {str(e)}")
            logger.debug(traceback.format_exc())
            
        # Final summary
        duration = time.time() - start_time
        logger.info(f"Batch processing complete: {processed_files}/{total_files} files in {duration:.3f}s")
        print("Converted segmentation masks to YOLOv8 format!")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        print("\nProcess interrupted by user")
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}")
        logger.critical(traceback.format_exc())
        print(f"An error occurred: {str(e)}")