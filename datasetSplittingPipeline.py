import os
import shutil
from sklearn.model_selection import train_test_split

def validate_and_organize_data(image_dir, label_dir, output_base_dir, val_size=0.15, random_state=42):
    """
    Validates, organizes, and renames images and labels for YOLOv8 training.

    Args:
        image_dir: Path to the directory containing images.
        label_dir: Path to the directory containing labels.
        output_base_dir: Base directory for the organized data.
        val_size: Proportion of data to use for validation (default: 0.15).
        random_state: Random seed for train-test split (default: 42).

    Returns:
        True if validation and organization were successful, False otherwise.
    """
    # Create output directories
    images_train_dir = os.path.join(output_base_dir, "images", "train")
    images_val_dir = os.path.join(output_base_dir, "images", "val")
    labels_train_dir = os.path.join(output_base_dir, "labels", "train")
    labels_val_dir = os.path.join(output_base_dir, "labels", "val")

    for dir_path in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Get files from directories
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    label_files = [f for f in os.listdir(label_dir) if f.lower().endswith('.txt')]

    # Match files based on the first 6 characters of their names
    matched_images = []
    matched_labels = []
    label_dict = {f[:6]: f for f in label_files}

    for img in image_files:
        prefix = img[:6]
        if prefix in label_dict:
            matched_images.append(img)
            matched_labels.append(label_dict[prefix])

    # Check for mismatches
    if len(matched_images) != len(matched_labels):
        print(f"Number of matched images ({len(matched_images)}) does not match number of matched labels ({len(matched_labels)})")
        return False

    # Split into train and val sets
    train_images, val_images, train_labels, val_labels = train_test_split(
        matched_images, matched_labels, test_size=val_size, random_state=random_state
    )

    # Copy and rename files with indexing
    copy_and_rename_files(train_images, train_labels, image_dir, label_dir, images_train_dir, labels_train_dir)
    copy_and_rename_files(val_images, val_labels, image_dir, label_dir, images_val_dir, labels_val_dir)

    print(f"Data split complete.")
    print(f"Train: {len(train_images)} images and {len(train_labels)} labels.")
    print(f"Val: {len(val_images)} images and {len(val_labels)} labels.")

    return True

def copy_and_rename_files(image_files, label_files, image_src_dir, label_src_dir, image_dest_dir, label_dest_dir):
    for index, (image_file, label_file) in enumerate(zip(image_files, label_files), start=1):
        # Create zero-padded filenames
        new_name = f"{index:07d}"

        # Copy and rename image
        image_ext = os.path.splitext(image_file)[1]
        shutil.copy(os.path.join(image_src_dir, image_file), os.path.join(image_dest_dir, f"{new_name}{image_ext}"))

        # Copy and rename label
        label_ext = os.path.splitext(label_file)[1]
        shutil.copy(os.path.join(label_src_dir, label_file), os.path.join(label_dest_dir, f"{new_name}{label_ext}"))

image_dir = '/content/myData/images'
label_dir = '/content/labels'
output_base_dir = '/content/train_data'

if validate_and_organize_data(image_dir, label_dir, output_base_dir):
    print("Indexing validation successful and data organized.")