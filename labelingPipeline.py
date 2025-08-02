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

# Clothing categories and mapping
LOWER_CLOTHES = {"pants", "shorts", "skirt", "tights", "stockings", "leg warmer"}
UPPER_CLOTHES = {"shirt", "blouse", "top", "t-shirt", "sweatshirt", "sweater", "cardigan",
                 "jacket", "vest", "coat", "dress", "jumpsuit", "cape", "tie",
                 "scarf", "hoodie", "collar", "sleeve"}
EXCLUDED_LABELS = {"shoe", "neckline", "belt"}
CLASS_NAMES = {0: "upper_clothes", 1: "lower_clothes", 2: "dress"}
CATEGORY_MAP = {"upper_clothes": "upper clothing",
                "lower_clothes": "lower clothing",
                "dress": "dress"}

class ClothingSegmenter:
    def __init__(self, model_path, images_output_dir):
        """Initialize the clothing segmentation models."""
        logger.info("Initializing ClothingSegmenter")
        start_time = time.time()
        self.images_output_dir = images_output_dir

        os.makedirs(self.images_output_dir, exist_ok=True)
        logger.info(f"Output directory created/confirmed: {self.images_output_dir}")

        # Load YOLOS model for clothing detection
        try:
            logger.info("Loading YOLOS model for clothing detection")
            self.processor = YolosImageProcessor.from_pretrained("valentinafeve/yolos-fashionpedia")
            self.model_yolos = YolosForObjectDetection.from_pretrained("valentinafeve/yolos-fashionpedia")
            logger.info("YOLOS model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading YOLOS model: {e}")
            raise

        # Load segmentation model
        try:
            logger.info(f"Loading YOLOv8 segmentation model from {model_path}")
            self.model_seg = YOLO(model_path)
            logger.info("YOLOv8 segmentation model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading YOLOv8 segmentation model: {e}")
            raise

        logger.info(f"Initialization completed in {time.time() - start_time:.2f} seconds")

    def keep_largest_connected_component(self, mask):
        mask = mask.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return mask
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        return (labels == largest_label).astype(np.uint8)

    def apply_crf(self, soft_mask, image, iterations=5):
        h, w = soft_mask.shape
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

    def get_dominant_colors(self, image, mask, k=3):
        region = image[mask == 1]
        if region.size == 0:
            return []
        non_white = region[np.all(region < 230, axis=1)]
        if non_white.size == 0:
            return []
        data = np.float32(non_white)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = centers.astype(int)
        counts = np.bincount(labels.flatten())
        idxs = np.argsort(-counts)
        colors = ['#{:02x}{:02x}{:02x}'.format(c[2], c[1], c[0]) for c in centers[idxs]]
        return colors

    def process_image(self, image_path, index):
        start = time.time()
        filename = os.path.basename(image_path)
        logger.info(f"Processing image {index:02d}: {filename}")

        orig = cv2.imread(image_path)
        if orig is None:
            logger.error(f"Cannot read image {image_path}")
            return

        # Background removal
        try:
            with open(image_path, 'rb') as f:
                inp = f.read()
            rem = remove(inp)
            arr = np.frombuffer(rem, np.uint8)
            bg_removed = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(bg_removed, cv2.COLOR_BGR2GRAY)
            _, bin_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            bg_removed = cv2.bitwise_and(bg_removed, bg_removed, mask=bin_mask)
        except Exception:
            logger.error("Background removal failed", exc_info=True)
            return

        # Initial segmentation for dress
        results = self.model_seg.predict(bg_removed, conf=0.5, iou=0.5)
        is_dress = False
        if results and results[0].boxes is not None:
            for cid in results[0].boxes.cls:
                if CLASS_NAMES.get(int(cid.item()), '').lower() == 'dress':
                    is_dress = True
                    break
        if is_dress:
            mask_t = results[0].masks.data[0].cpu().numpy().astype(np.float32)
            soft = cv2.resize(mask_t, (bg_removed.shape[1], bg_removed.shape[0]), interpolation=cv2.INTER_NEAREST)
            soft = np.clip(soft, 0, 1)
            refined = self.apply_crf(soft, bg_removed)
            smooth = cv2.GaussianBlur(refined.astype(np.float32), (17, 17), 0)
            clean = (smooth > 0.5).astype(np.uint8)
            clean = self.keep_largest_connected_component(clean)
            seg = np.where(clean[:, :, None] == 1, bg_removed, 255)
            out_fn = f"{index:02d}_dress.jpg"
            cv2.imwrite(os.path.join(self.images_output_dir, out_fn), seg)
            colors = self.get_dominant_colors(bg_removed, clean)
            meta = {"filename": out_fn, "category": "dress", "colors": colors, "description": "dress"}
            with open(os.path.join(self.images_output_dir, f"{index:02d}_dress.json"), 'w') as mf:
                json.dump(meta, mf, indent=2)
            logger.info(f"Saved dress segmentation: {out_fn}")
            return

        # YOLOS detection
        try:
            pil = Image.fromarray(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
            inp = self.processor(images=pil, return_tensors="pt")
            with torch.no_grad():
                out = self.model_yolos(**inp)
            targets = torch.tensor([pil.size[::-1]])
            res = self.processor.post_process_object_detection(out, threshold=0.9, target_sizes=targets)[0]
            lower_labels, upper_labels = set(), set()
            for label in res['labels']:
                name = self.model_yolos.config.id2label[label.item()]
                if name in LOWER_CLOTHES:
                    lower_labels.add(name)
                elif name in UPPER_CLOTHES:
                    upper_labels.add(name)
            logger.info(f"YOLOS lower: {lower_labels or 'None'} | upper: {upper_labels or 'None'}")
        except Exception:
            logger.error("YOLOS detection failed", exc_info=True)

        # Full segmentation pass
        try:
            results = self.model_seg.predict(bg_removed, conf=0.5, iou=0.5)
            if not results or not results[0].masks:
                logger.warning("No masks for full segmentation")
                return
            for i, (mask_t, cid) in enumerate(zip(results[0].masks.data, results[0].boxes.cls)):
                cid = int(cid.item())
                mask_t = mask_t.cpu().numpy().astype(np.float32)
                soft = cv2.resize(mask_t, (bg_removed.shape[1], bg_removed.shape[0]), interpolation=cv2.INTER_NEAREST)
                soft = np.clip(soft, 0, 1)
                refined = self.apply_crf(soft, bg_removed)
                smooth = cv2.GaussianBlur(refined.astype(np.float32), (17, 17), 0)
                clean = (smooth > 0.5).astype(np.uint8)
                clean = self.keep_largest_connected_component(clean)
                seg = np.where(clean[:, :, None] == 1, bg_removed, 255)
                out_fn = f"{index:02d}_{cid}.jpg"
                cv2.imwrite(os.path.join(self.images_output_dir, out_fn), seg)
                colors = self.get_dominant_colors(bg_removed, clean)
                desc = ' '.join(lower_labels) if CLASS_NAMES.get(cid)=='lower_clothes' else (' '.join(upper_labels) if CLASS_NAMES.get(cid)=='upper_clothes' else '')
                meta = {"filename": out_fn, "category": CATEGORY_MAP.get(CLASS_NAMES.get(cid,''), ''), "colors": colors, "description": desc}
                with open(os.path.join(self.images_output_dir, f"{index:02d}_{cid}.json"), 'w') as mf:
                    json.dump(meta, mf, indent=2)
                logger.info(f"Saved {out_fn} (class {CLASS_NAMES.get(cid)})")
        except Exception:
            logger.error("Full segmentation failed", exc_info=True)

        logger.info(f"Completed {filename} in {time.time() - start:.2f}s")

    def process_directory(self, input_dir):
        logger.info(f"Processing directory: {input_dir}")
        files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg','.jpeg','.png','.webp'))])
        if not files:
            logger.warning("No images found")
            return
        for idx, fn in enumerate(files):
            self.process_image(os.path.join(input_dir, fn), idx)
        logger.info("Batch processing complete")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='../input_images')
    parser.add_argument('--output_dir', type=str, default='../output_images')
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Input: {args.input_dir} | Output: {args.output_dir} | Model: {args.model_path}")

    segmenter = ClothingSegmenter(args.model_path, args.output_dir)
    segmenter.process_directory(args.input_dir)
