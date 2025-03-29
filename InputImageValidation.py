import cv2
import numpy as np
import logging
import argparse
from ultralytics import YOLO

def isImageValid(input_path: str) -> str:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Define the required keypoint indices (COCO ordering):
    # Shoulders (5,6), wrists (9,10), hips (11,12), knees (13,14), ankles (15,16)
    required_indices = [5, 6, 9, 10, 11, 12, 13, 14, 15, 16]
    # Margin in pixels to decide if a keypoint is too close to the image boundary
    margin = 5

    # Load the pretrained YOLOv8 pose model
    try:
        model = YOLO("yolov8m-pose.pt")
    except Exception as e:
        logging.error(f"Error loading YOLO model: {e}")
        raise

    # Read the input image
    img = cv2.imread(input_path)
    if img is None:
        logging.error(f"Could not load image at {input_path}")
        raise FileNotFoundError(f"Could not load image at {input_path}")
    else:
        logging.info(f"Loaded image {input_path} with shape {img.shape}")

    img_h, img_w = img.shape[:2]

    # Run pose estimation
    try:
        results = model(img)
    except Exception as e:
        logging.error(f"Error during pose estimation: {e}")
        raise

    # Check each detected person's keypoints for completeness.
    # If at least one person with a complete pose is found, the image is considered valid.
    is_valid = False
    for result in results:
        if result.keypoints is None:
            logging.warning("No keypoints detected for one of the persons. Skipping.")
            continue

        for person in result.keypoints.xy:
            complete_pose = True

            # Check each required keypoint is present and not too close to the image borders.
            for kp_idx in required_indices:
                if kp_idx >= len(person):
                    logging.warning("Incomplete keypoint data detected.")
                    complete_pose = False
                    break

                x, y = person[kp_idx]
                if (x < margin or y < margin or x > (img_w - margin) or y > (img_h - margin)):
                    complete_pose = False
                    break

            if complete_pose:
                is_valid = True
                break  # Found a valid pose for one person, no need to check further

        if is_valid:
            break

    # Output the result: empty string if valid, warning message if not.
    if is_valid:
        output = ""
        logging.info("Valid pose detected.")
    else:
        output = "Incomplete pose detected: To obtain the most accurate results, please capture an image showing your full body"
        logging.warning(output)
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check if the input image has a complete pose")
    parser.add_argument("image_path", help="Path to the input image")
    args = parser.parse_args()

    result = isImageValid(args.image_path)
    print(result)
