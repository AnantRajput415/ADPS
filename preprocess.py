import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Paths
image_dir = "car_racing_images"  # Directory with raw images
label_file = "car_racing_labels.txt"  # File with labels
output_image_dir = "preprocessed_images"
output_label_file = "preprocessed_labels.csv"

# Parameters
target_size = (96, 96)  # Resize images to this resolution
os.makedirs(output_image_dir, exist_ok=True)

# Read labels
labels = pd.read_csv(label_file)

# List to store normalized bounding boxes
normalized_labels = []

# Preprocessing loop
for _, row in tqdm(labels.iterrows(), total=len(labels)):
    image_name = row["image_name"]
    car_x, car_y, car_w, car_h = row[" car_x"], row[" car_y"], row[" car_w"], row[" car_h"]
    road_x, road_y, road_w, road_h = row[" road_x"], row[" road_y"], row[" road_w"], row[" road_h"]

    # Load image
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Image {image_name} could not be loaded.")
        continue

    # Get original dimensions
    orig_height, orig_width = image.shape[:2]

    # Resize image
    resized_image = cv2.resize(image, target_size)

    # Save preprocessed image
    output_image_path = os.path.join(output_image_dir, image_name)
    cv2.imwrite(output_image_path, resized_image)

    # Normalize bounding box coordinates
    car_bbox = [
        car_x / orig_width,
        car_y / orig_height,
        car_w / orig_width,
        car_h / orig_height,
    ]
    road_bbox = [
        road_x / orig_width,
        road_y / orig_height,
        road_w / orig_width,
        road_h / orig_height,
    ]

    # Combine and save
    normalized_labels.append([image_name, *car_bbox, *road_bbox])

# Save normalized labels
columns = [
    "image_name",
    "car_x_norm", "car_y_norm", "car_w_norm", "car_h_norm",
    "road_x_norm", "road_y_norm", "road_w_norm", "road_h_norm",
]
normalized_labels_df = pd.DataFrame(normalized_labels, columns=columns)
normalized_labels_df.to_csv(output_label_file, index=False)

print(f"Preprocessing completed. Preprocessed images saved in '{output_image_dir}' and labels in '{output_label_file}'.")
