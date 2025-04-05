import os
import cv2
import numpy as np
import json
import logging
import time
import traceback
from ultralytics import YOLO
from rembg import remove
import torch
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import argparse
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("clothing_segmentation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Filter out specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*The parameter 'pretrained' is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*You seem to be using the pipelines sequentially on GPU.*")

# Updated clothing categories and label mapping
LOWER_CLOTHES = {"pants", "shorts", "skirt", "tights", "stockings", "leg warmer"}
UPPER_CLOTHES = {"shirt", "blouse", "top", "t-shirt", "sweatshirt", "sweater", "cardigan",
                 "jacket", "vest", "coat", "dress", "jumpsuit", "cape", "tie",
                 "scarf", "hoodie", "collar", "sleeve"}
EXCLUDED_LABELS = {"shoe", "neckline", "belt"}
# Adjust CLASS_NAMES: Assuming the segmentation model now supports three classes.
CLASS_NAMES = {0: "upper_clothes", 1: "lower_clothes", 2: "dress"}
CATEGORY_MAP = {
    "upper_clothes": "upper clothing",
    "lower_clothes": "lower clothing",
    "dress": "dress"
}

class ClothingSegmenter:
    def __init__(self, model_path, images_output_dir):
        """Initialize the clothing segmentation models."""
        logger.info("Initializing ClothingSegmenter")
        start_time = time.time()
        self.images_output_dir = images_output_dir

        try:
            os.makedirs(self.images_output_dir, exist_ok=True)
            logger.info(f"Output directory created/confirmed: {self.images_output_dir}")
            
            # Load YOLOS model for clothing detection
            try:
                logger.info("Loading YOLOS model for clothing detection")
                self.processor = YolosImageProcessor.from_pretrained("valentinafeve/yolos-fashionpedia")
                self.model_yolos = YolosForObjectDetection.from_pretrained("valentinafeve/yolos-fashionpedia")
                logger.info("YOLOS model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading YOLOS model: {str(e)}")
                raise RuntimeError(f"Failed to load YOLOS model: {str(e)}")
            
            # Load segmentation model
            try:
                logger.info(f"Loading YOLOv8 segmentation model from {model_path}")
                self.model_seg = YOLO(model_path)
                logger.info("YOLOv8 segmentation model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading YOLOv8 segmentation model: {str(e)}")
                raise RuntimeError(f"Failed to load YOLOv8 model: {str(e)}")
                
            logger.info(f"Initialization completed in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.critical(f"Critical error during initialization: {str(e)}")
            logger.debug(traceback.format_exc())
            raise

    def keep_largest_connected_component(self, mask):
        """Keep only the largest connected component in a binary mask."""
        try:
            mask = mask.astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if num_labels <= 1:
                return mask
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            return np.where(labels == largest_label, 1, 0).astype(np.uint8)
        except Exception as e:
            logger.error(f"Error in keep_largest_connected_component: {str(e)}")
            return mask  # Return original mask on error

    def apply_crf(self, soft_mask, image, iterations=5):
        """
        Apply DenseCRF to refine the segmentation mask.
        """
        try:
            h, w = soft_mask.shape
            # Prepare probability map: 2 channels: background and foreground.
            probs = np.stack([1 - soft_mask, soft_mask], axis=0)
            d = dcrf.DenseCRF2D(w, h, 2)
            U = unary_from_softmax(probs.astype(np.float32))
            d.setUnaryEnergy(U)
            d.addPairwiseGaussian(sxy=5, compat=5)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            d.addPairwiseBilateral(sxy=60, srgb=10, rgbim=image_rgb, compat=15)
            Q = d.inference(iterations)
            refined_mask = np.argmax(np.array(Q).reshape((2, h, w)), axis=0).astype(np.uint8)
            return refined_mask
        except Exception as e:
            logger.error(f"Error in apply_crf: {str(e)}")
            return np.where(soft_mask > 0.5, 1, 0).astype(np.uint8)

    def get_dominant_colors(self, image, mask, k=3):
        """Extract dominant colors from the masked region of an image."""
        try:
            region_pixels = image[mask == 1]
            if region_pixels.size == 0:
                logger.warning("No pixels found in masked region for color extraction")
                return []
            non_white = region_pixels[np.all(region_pixels < 230, axis=1)]
            if non_white.size == 0:
                logger.warning("No non-white pixels found in masked region")
                return []
            data = np.float32(non_white)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            attempts = 10
            ret, labels, centers = cv2.kmeans(data, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
            centers = centers.astype(int)
            counts = np.bincount(labels.flatten())
            sorted_idx = np.argsort(-counts)
            dominant_colors = []
            for idx in sorted_idx:
                bgr_color = centers[idx]
                rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
                hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb_color)
                dominant_colors.append(hex_color)
            return dominant_colors
        except Exception as e:
            logger.error(f"Error in get_dominant_colors: {str(e)}")
            return []

    def process_image(self, image_path, index):
        """Process a single image to detect and segment clothing items."""
        start_time = time.time()
        filename = os.path.basename(image_path)
        logger.info(f"Processing image {index:02d}: {filename}")
        
        try:
            orig_img = cv2.imread(image_path)
            if orig_img is None:
                logger.error(f"Unable to read image: {image_path}")
                return
            
            # Background removal
            try:
                logger.debug("Removing background")
                with open(image_path, "rb") as file:
                    input_bytes = file.read()
                rembg_output = remove(input_bytes)
                data = np.frombuffer(rembg_output, np.uint8)
                bg_removed_img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if bg_removed_img is None:
                    logger.error("Failed to decode background-removed image")
                    return
                gray = cv2.cvtColor(bg_removed_img, cv2.COLOR_BGR2GRAY)
                _, binary_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
                bg_removed_img = cv2.bitwise_and(bg_removed_img, bg_removed_img, mask=binary_mask)
            except Exception as e:
                logger.error(f"Error during background removal: {str(e)}")
                logger.debug(traceback.format_exc())
                return

            # Initial segmentation to check for dress
            try:
                logger.debug("Running YOLOv8 segmentation for dress detection")
                results = self.model_seg.predict(bg_removed_img, conf=0.5, iou=0.5)
                is_dress = False
                if results and results[0].boxes is not None:
                    for class_id_tensor in results[0].boxes.cls:
                        class_id = int(class_id_tensor.item())
                        if CLASS_NAMES.get(class_id, "").lower() == "dress":
                            is_dress = True
                            break

                if is_dress:
                    logger.info("Detected dress - processing as single item")
                    mask_tensor = results[0].masks.data[0]
                    mask_np = mask_tensor.cpu().numpy().astype(np.float32)
                    soft_mask = cv2.resize(mask_np, (bg_removed_img.shape[1], bg_removed_img.shape[0]), 
                                             interpolation=cv2.INTER_NEAREST)
                    soft_mask = np.clip(soft_mask, 0, 1)
                    refined_mask = self.apply_crf(soft_mask, bg_removed_img)
                    smoothed_mask = cv2.GaussianBlur(refined_mask.astype(np.float32), (17, 17), 0)
                    smoothed_mask = np.where(smoothed_mask > 0.5, 1, 0).astype(np.uint8)
                    cleaned_mask = self.keep_largest_connected_component(smoothed_mask)
                    segmented_output = np.where(cleaned_mask[:, :, None] == 1, bg_removed_img, 255)
                    colors = self.get_dominant_colors(bg_removed_img, cleaned_mask)
                    
                    seg_filename = f"{index:02d}_dress.jpg"
                    seg_output_path = os.path.join(self.images_output_dir, seg_filename)
                    cv2.imwrite(seg_output_path, segmented_output)
                    
                    metadata = {
                        "filename": seg_filename,
                        "category": "dress",
                        "colors": colors if colors else [],
                        "description": "dress"
                    }
                    metadata_filename = f"{index:02d}_dress.json"
                    metadata_path = os.path.join(self.images_output_dir, metadata_filename)
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    logger.info(f"Saved dress segmentation as {seg_filename}")
                    return  # Exit early if dress is detected

<<<<<<< HEAD
                # --- YOLOS detection for additional info ---
                # Initialize the label sets so they are available later.
                lower_clothes_labels = set()
                upper_clothes_labels = set()
=======
                # YOLOS clothing detection for additional info
>>>>>>> ae2b199280368cecfce92b4fcc87fa3a5b3f3e5f
                try:
                    pil_image = Image.fromarray(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
                    inputs_yolos = self.processor(images=pil_image, return_tensors="pt")
                    with torch.no_grad():
                        outputs_yolos = self.model_yolos(**inputs_yolos)
                    target_sizes = torch.tensor([pil_image.size[::-1]])
                    results_yolos = self.processor.post_process_object_detection(
                        outputs_yolos, threshold=0.9, target_sizes=target_sizes)[0]
                    
<<<<<<< HEAD
=======
                    lower_clothes_labels = set()
                    upper_clothes_labels = set()
>>>>>>> ae2b199280368cecfce92b4fcc87fa3a5b3f3e5f
                    for label in results_yolos["labels"]:
                        label_name = (self.model_yolos.config.id2label[label.item()]
                                      if self.model_yolos.config.id2label else str(label.item()))
                        if label_name in LOWER_CLOTHES:
                            lower_clothes_labels.add(label_name)
                        elif label_name in UPPER_CLOTHES:
                            upper_clothes_labels.add(label_name)
                    
<<<<<<< HEAD
                    logger.info(f"YOLOS detected lower clothing: {' '.join(lower_clothes_labels) if lower_clothes_labels else 'None'}")
                    logger.info(f"YOLOS detected upper clothing: {' '.join(upper_clothes_labels) if upper_clothes_labels else 'None'}")
=======
                    logger.info(f"YOLOS detected lower clothing: {' '.join(lower_clothes_labels)}")
                    logger.info(f"YOLOS detected upper clothing: {' '.join(upper_clothes_labels)}")
>>>>>>> ae2b199280368cecfce92b4fcc87fa3a5b3f3e5f
                except Exception as e:
                    logger.error(f"Error during YOLOS detection: {str(e)}")
                    logger.debug(traceback.format_exc())
                
<<<<<<< HEAD
                # --- Segmentation for each detected mask for non-dress items ---
=======
                # Perform segmentation for each detected mask
>>>>>>> ae2b199280368cecfce92b4fcc87fa3a5b3f3e5f
                try:
                    logger.debug("Running YOLOv8 segmentation for full processing")
                    results = self.model_seg.predict(bg_removed_img, conf=0.5, iou=0.5)
                    if not (results and results[0].masks and results[0].masks.data is not None):
                        logger.warning("No segmentation masks detected")
                        return
                    
                    for i, (mask_tensor, class_id_tensor) in enumerate(zip(results[0].masks.data, results[0].boxes.cls)):
                        class_id = int(class_id_tensor.item())
                        mask_np = mask_tensor.cpu().numpy().astype(np.float32)
                        soft_mask = cv2.resize(mask_np, (bg_removed_img.shape[1], bg_removed_img.shape[0]),
                                                 interpolation=cv2.INTER_NEAREST)
                        soft_mask = np.clip(soft_mask, 0, 1)
                        logger.debug(f"Applying DenseCRF refinement for mask {i+1}")
                        refined_mask = self.apply_crf(soft_mask, bg_removed_img, iterations=5)
                        smoothed_mask = cv2.GaussianBlur(refined_mask.astype(np.float32), (17, 17), 0)
                        smoothed_mask = np.where(smoothed_mask > 0.5, 1, 0).astype(np.uint8)
                        cleaned_mask = self.keep_largest_connected_component(smoothed_mask)
                        segmented_output = np.where(cleaned_mask[:, :, None] == 1, bg_removed_img, 255)
                        
                        seg_filename = f"{index:02d}_{class_id}.jpg"
                        seg_output_path = os.path.join(self.images_output_dir, seg_filename)
                        cv2.imwrite(seg_output_path, segmented_output)
                        logger.info(f"Saved segmentation as {seg_filename} (class: {CLASS_NAMES.get(class_id, class_id)})")
                        
                        colors = self.get_dominant_colors(bg_removed_img, cleaned_mask)
<<<<<<< HEAD
                        
                        # Create metadata JSON file
                        if CLASS_NAMES.get(class_id, "").lower() == "dress":
                            # For dress, description is always "dress"
                            description = "dress"
                        elif CLASS_NAMES.get(class_id, "").lower() == "lower_clothes":
                            description = " ".join(lower_clothes_labels) if lower_clothes_labels else ""
                        elif CLASS_NAMES.get(class_id, "").lower() == "upper_clothes":
                            description = " ".join(upper_clothes_labels) if upper_clothes_labels else ""
                        else:
                            description = ""
                        
                        metadata = {
                            "filename": seg_filename,
                            "category": CATEGORY_MAP.get(CLASS_NAMES.get(class_id, ""), ""),
                            "colors": colors if colors else [],
                            "description": description
                        }
                        
                        metadata_filename = f"{index:02d}_{class_id}.json"
                        metadata_path = os.path.join(self.images_output_dir, metadata_filename)
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata, f, indent=2)
                        logger.info(f"Saved metadata file {metadata_filename}")
=======
                        if colors:
                            logger.debug(f"Dominant colors: {colors}")
>>>>>>> ae2b199280368cecfce92b4fcc87fa3a5b3f3e5f
                except Exception as e:
                    logger.error(f"Error during segmentation processing: {str(e)}")
                    logger.debug(traceback.format_exc())
                    
                logger.info(f"Completed processing {filename} in {time.time() - start_time:.2f} seconds")
            except Exception as e:
                logger.error(f"Unhandled error processing {filename}: {str(e)}")
                logger.debug(traceback.format_exc())
        except Exception as e:
            logger.error(f"Unhandled error processing {filename}: {str(e)}")
            logger.debug(traceback.format_exc())

    def process_directory(self, input_dir):
        """Process all images in the input directory sequentially."""
        start_time = time.time()
        logger.info(f"Starting batch processing of images from {input_dir}")
        
        try:
            image_files = sorted([f for f in os.listdir(input_dir)
                                  if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))])
            
            if not image_files:
                logger.warning(f"No images found in {input_dir}")
                return
            
            logger.info(f"Found {len(image_files)} images to process")
            
            for idx, filename in enumerate(image_files):
                image_path = os.path.join(input_dir, filename)
                self.process_image(image_path, idx)
            
            logger.info(f"Batch processing completed in {time.time() - start_time:.2f} seconds")
            logger.info(f"Successfully processed {len(image_files)} images")
        except Exception as e:
            logger.critical(f"Critical error during batch processing: {str(e)}")
            logger.debug(traceback.format_exc())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clothing Segmentation Script")
    parser.add_argument("--input_dir", type=str, default="/content/input_images", help="Directory with input images")
    parser.add_argument("--output_dir", type=str, default="/content/test_data/images", help="Directory to save outputs")
    parser.add_argument("--model_path", type=str, default="/content/drive/MyDrive/OUSTORA_ONLY/yolov8_large_experiment_extended/weights/best.pt", help="Path to YOLOv8 segmentation model")
    args = parser.parse_args()

    try:
        logger.info("=== Starting Clothing Segmentation Script ===")
        logger.info(f"Input Directory: {args.input_dir}")
        logger.info(f"Output Directory: {args.output_dir}")
        logger.info(f"Model Path: {args.model_path}")

<<<<<<< HEAD
        # Use the directory part of output_dir as images_output_dir if needed
=======
>>>>>>> ae2b199280368cecfce92b4fcc87fa3a5b3f3e5f
        segmenter = ClothingSegmenter(args.model_path, os.path.dirname(args.output_dir))
        segmenter.process_directory(args.input_dir)
        
        logger.info("=== Script execution completed successfully ===")
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        print("\nProcess interrupted by user")
    except Exception as e:
        logger.critical(f"Unhandled exception in main process: {str(e)}")
        logger.critical(traceback.format_exc())
        print(f"An error occurred: {str(e)}")
        print("Check the log file for details.")
