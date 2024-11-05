import cv2
import numpy as np
import random
import os

# Function to apply a random perspective transformation
def perspective_transform(image):
    h, w = image.shape[:2]
    # Define random points for perspective shift
    margin = int(min(h, w) * 0.1)
    pts1 = np.float32([
        [random.randint(0, margin), random.randint(0, margin)], 
        [w - random.randint(0, margin), random.randint(0, margin)], 
        [random.randint(0, margin), h - random.randint(0, margin)], 
        [w - random.randint(0, margin), h - random.randint(0, margin)]
    ])
    pts2 = np.float32([
        [0, 0], [w, 0], [0, h], [w, h]
    ])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, matrix, (w, h))

# Function to adjust brightness
def adjust_brightness(image, value=30):
    image = cv2.convertScaleAbs(image, alpha=1, beta=value)
    return image

# Function to adjust contrast
def adjust_contrast(image, factor=1.5):
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

# Function to flip the image vertically
def flip_image(image):
    return cv2.flip(image, 0)

# Main function to perform a random augmentation
def random_augment_image(image):
    # List of augmentation functions without rotation
    augmentations = [
        perspective_transform,                                                 # Random perspective
        lambda img: adjust_brightness(img, random.randint(-30, 30)),           # Brightness adjustment
        lambda img: adjust_contrast(img, random.uniform(0.8, 1.2)),            # Contrast adjustment
        flip_image                                                             # Vertical flip
    ]
    
    # Select a random augmentation
    aug_func = random.choice(augmentations)
    return aug_func(image)

# Specify the folder for augmentation
input_folder = "./data/new_pool/REVERSE"
output_folder = "./data/augmented_pool/REVERSE"

os.makedirs(output_folder, exist_ok=True)
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

for filename in os.listdir(input_folder):
    if filename.lower().endswith(valid_extensions):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        
        if image is not None:
            augmented_image = random_augment_image(image)
            output_path = os.path.join(output_folder, f"{filename.split('.')[0]}_aug.png")
            cv2.imwrite(output_path, augmented_image)
