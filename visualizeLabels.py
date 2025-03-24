import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def visualize_label(image_path, label_path, image_size=(768, 1024), class_colors=None):
    """
    Visualizes a segmentation mask with improved overlay and class color support.

    Args:
        image_path: Path to the image file.
        label_path: Path to the label file (YOLOv8 segmentation format).
        image_size: Tuple containing the expected image dimensions (height, width).
        class_colors: Dictionary mapping class labels to RGB colors.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        resized_image = cv2.resize(image, (image_size[1], image_size[0]))
        mask = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)

        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split()
                class_label = int(data[0])
                polygon = [
                    (int(float(data[i]) * image_size[1]), int(float(data[i + 1]) * image_size[0]))
                    for i in range(1, len(data), 2)
                ]
                pts = np.array(polygon, dtype=np.int32)
                color = class_colors.get(class_label, (0, 255, 0))  # Default color if not specified
                cv2.fillPoly(mask, [pts], color)

        overlay = cv2.addWeighted(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB), 0.5, mask, 0.5, 0)

        plt.figure(figsize=(10, 6))
        plt.imshow(overlay)
        plt.axis('off')
        plt.show()

    except Exception as e:
        print(f"Error visualizing {image_path}: {e}")

image_path = '050184_4.png'
label_path = '050184_4.txt'
class_colors = {
    0: (255, 0, 0),
    1: (0, 255, 0)
}
visualize_label(image_path, label_path, class_colors=class_colors)
