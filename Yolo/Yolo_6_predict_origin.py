# -------------------------------------------------------------------------------------------------
# * @author  21900727 Garam Jin & 22100034 Eunji Ko
# * @Date    2024-06-24
# * @Mod	 2024-06-10 by YKKIM
# * @brief   Final Project(DLIP)
# -------------------------------------------------------------------------------------------------

import os
import random
import cv2 as cv
import numpy as np
from ultralytics import YOLO
from matplotlib import pyplot as plt

# Custom Dataset Class
class ImageDataset:
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = cv.imread(img_path)
        mask_path = self.mask_files[idx]
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE) if mask_path else np.zeros_like(image[:, :, 0])
        return img_path, image, mask

    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.mask_files = [self.get_mask_path(f) for f in self.image_files]

    def get_mask_path(self, image_path):
        base_name = os.path.basename(image_path)
        mask_name = base_name.replace('.jpg', '_mask.png').replace('.jpeg', '_mask.png').replace('.png', '_mask.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        return mask_path if os.path.exists(mask_path) else None

    def __len__(self):
        return len(self.image_files)

# Image and mask directory paths
image_dir = 'C:/Users/ehrpr/source/repos/DLIP/Datasets/val/images'
mask_dir = 'C:/Users/ehrpr/source/repos/DLIP/Datasets/val/masks_mask'

# Datasets and Dataloaders
image_dataset = ImageDataset(image_dir=image_dir, mask_dir=mask_dir)

# YOLO model load
model = YOLO('C:/Users/ehrpr/source/repos/DLIP/Datasets/runs/segment/train26_seg_n_30/weights/best.pt')

# Select random images for visualization
num_images_to_show = 6
random_indices = random.sample(range(len(image_dataset)), num_images_to_show)

# Select random images
images_to_show = [image_dataset[i] for i in random_indices]

def visualize_results(images, model):
    fig, axs = plt.subplots(num_images_to_show, 3, figsize=(12, 12))

    for i, (img_path, image, mask) in enumerate(images):
        # Predict using YOLO model
        result = model.predict(source=image, save=False)
        r = result[0]
        dst = r.plot()

        # Original image
        axs[i, 0].imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        axs[i, 0].axis('off')
        if i == 0:
            axs[i, 0].set_title('Original Image')

        # Ground truth mask
        axs[i, 1].imshow(mask, cmap='gray')
        axs[i, 1].axis('off')
        if i == 0:
            axs[i, 1].set_title('Ground Truth Mask')

        # Predicted mask
        axs[i, 2].imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
        axs[i, 2].axis('off')
        if i == 0:
            axs[i, 2].set_title('Predicted Mask')

    plt.tight_layout()
    plt.show()

# Visualize results
visualize_results(images_to_show, model)
