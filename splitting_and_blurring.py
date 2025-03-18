import os
import cv2
import numpy as np
import random
import shutil
from tqdm import tqdm

def get_all_images(input_folder):
    image_paths = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(root, file))
    return image_paths

def apply_augmentations(image):
    augmented_images = []
    
    # Define filters
    kernel_sharpening = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
    
    for _ in range(5):  # Generate 5 blurred images
        ksize = random.choice([3, 5, 7, 9])  # Random kernel size
        blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
        augmented_images.append(blurred)
    
    for _ in range(5):  # Generate 5 sharpened images
        sharpened = cv2.filter2D(image, -1, kernel_sharpening)
        augmented_images.append(sharpened)
    
    return augmented_images

def process_images(input_folder, output_folder):
    image_paths = get_all_images(input_folder)
    random.shuffle(image_paths)
    
    train_split = int(0.8 * len(image_paths))
    train_images = image_paths[:train_split]
    val_images = image_paths[train_split:]
    
    for mode, images in zip(["train", "val"], [train_images, val_images]):
        for img_path in tqdm(images, desc=f"Processing {mode}"):
            rel_path = os.path.relpath(img_path, input_folder)
            new_dir = os.path.join(output_folder, mode, os.path.dirname(rel_path))
            os.makedirs(new_dir, exist_ok=True)
            
            image = cv2.imread(img_path)
            augmented_images = apply_augmentations(image)
            
            for idx, aug_img in enumerate(augmented_images):
                new_filename = os.path.join(new_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_{idx}.jpg")
                cv2.imwrite(new_filename, aug_img)

if __name__ == "__main__":
    input_folder = input("Enter the path to the input folder: ")
    output_folder = input("Enter the path to the output folder: ")
    os.makedirs(output_folder, exist_ok=True)
    
    process_images(input_folder, output_folder)
    print("Processing complete!")