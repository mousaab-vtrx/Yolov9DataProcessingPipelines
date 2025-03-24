import os
import cv2
import numpy as np
import json
from ultralytics import YOLO
from rembg import remove
import torch
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image

try:
    processor = YolosImageProcessor.from_pretrained("valentinafeve/yolos-fashionpedia")
    model_yolos = YolosForObjectDetection.from_pretrained("valentinafeve/yolos-fashionpedia")
except OSError as e:
    print(f"Error loading YOLOS model: {e}")
    exit(1)

model_seg = YOLO('/content/drive/MyDrive/OUSTORA_ONLY/yolov8_large_experiment_extended/weights/best.pt')


lower_clothes = [
    "pants", "shorts", "skirt", "tights, stockings", "leg warmer"
]
upper_clothes = [
    "shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan",
    "jacket", "vest", "coat", "dress", "jumpsuit", "cape", "tie",
    "scarf", "hoodie", "collar", "sleeve"
]
EXCLUDED_LABELS = ["shoe", "neckline", "belt"]

class_names = {0: "upper_clothes", 1: "lower_clothes"}
category_map = {
    "upper_clothes": "upper clothing",
    "lower_clothes": "lower clothing"
}

input_dir = "/content/input_images"
output_root = "/content/test_data"
images_output_dir = os.path.join(output_root, "images")
os.makedirs(images_output_dir, exist_ok=True)


def keep_largest_connected_component(mask):
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == largest_label, 1, 0).astype(np.uint8)

def apply_crf(soft_mask, image, iterations=5):
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

def get_dominant_colors(image, mask, k=3):
    region_pixels = image[mask == 1]
    if region_pixels.size == 0:
        return []
    
    non_white = region_pixels[np.all(region_pixels < 250, axis=1)]
    if non_white.size == 0:
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


def process_image(image_path, index):
    print(f"\nProcessing image {index:02d}: {image_path}")

    orig_img = cv2.imread(image_path)
    if orig_img is None:
        print("  [!] Unable to read image. Skipping.")
        return

    pil_image = Image.fromarray(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    inputs_yolos = processor(images=pil_image, return_tensors="pt")
    outputs_yolos = model_yolos(**inputs_yolos)
    target_sizes = torch.tensor([pil_image.size[::-1]])
    results_yolos = processor.post_process_object_detection(outputs_yolos, threshold=0.9, target_sizes=target_sizes)[0]

    lower_clothes_labels = set()
    upper_clothes_labels = set()
    for label in results_yolos["labels"]:
        label_name = (model_yolos.config.id2label[label.item()]
                      if model_yolos.config.id2label else str(label.item()))
        if label_name in lower_clothes:
            lower_clothes_labels.add(label_name)
        elif label_name in upper_clothes:
            upper_clothes_labels.add(label_name)
    
    lower_description = " ".join(lower_clothes_labels)
    upper_description = " ".join(upper_clothes_labels)
    print("  YOLOS detected lower clothing:", lower_description)
    print("  YOLOS detected upper clothing:", upper_description)

    with open(image_path, "rb") as file:
        input_bytes = file.read()
    rembg_output = remove(input_bytes)
    data = np.frombuffer(rembg_output, np.uint8)
    bg_removed_img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bg_removed_img is None:
        print("  [!] Failed to decode background-removed image. Skipping...")
        return

    gray = cv2.cvtColor(bg_removed_img, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    bg_removed_img = cv2.bitwise_and(bg_removed_img, bg_removed_img, mask=binary_mask)

    results = model_seg.predict(bg_removed_img, conf=0.5, iou=0.5)
    if not (results and results[0].masks and results[0].masks.data is not None):
        print("  [!] No masks detected.")
        return

    for mask_tensor, class_id_tensor in zip(results[0].masks.data, results[0].boxes.cls):
        class_id = int(class_id_tensor.item())
        mask_np = mask_tensor.cpu().numpy().astype(np.float32)
        soft_mask = cv2.resize(mask_np, (bg_removed_img.shape[1], bg_removed_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        soft_mask = np.clip(soft_mask, 0, 1)

        refined_mask = apply_crf(soft_mask, bg_removed_img, iterations=5)

        smoothed_mask = cv2.GaussianBlur(refined_mask.astype(np.float32), (17, 17), 0)
        smoothed_mask = np.where(smoothed_mask > 0.5, 1, 0).astype(np.uint8)

        cleaned_mask = keep_largest_connected_component(smoothed_mask)
        segmented_output = np.where(cleaned_mask[:, :, None] == 1, bg_removed_img, 255)

        seg_filename = f"{index:02d}_{class_id}.jpg"
        seg_output_path = os.path.join(images_output_dir, seg_filename)
        cv2.imwrite(seg_output_path, segmented_output)
        print(f"  Saved segmentation as {seg_output_path}")

if __name__ == "__main__":
    image_files = sorted([f for f in os.listdir(input_dir)
                          if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))])
    if not image_files:
        print(f"[!] No images found in {input_dir}.")
    for idx, filename in enumerate(image_files):
        process_image(os.path.join(input_dir, filename), idx)
