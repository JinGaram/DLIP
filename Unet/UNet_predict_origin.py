# -------------------------------------------------------------------------------------------------
# * @author  21900727 Garam Jin & 22100034 Eunji Ko
# * @Date    2024-06-24
# * @Mod	 2024-06-10 by YKKIM
# * @brief   Final Project(DLIP)
# -------------------------------------------------------------------------------------------------

import os
import random
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
import re

class SegmentationDataset:
    def __getitem__(self, idx):
        img_name, lbl_name = self.image_label_pairs[idx]
        img_path = os.path.join(self.image_dir, img_name)
        lbl_path = os.path.join(self.label_dir, lbl_name)

        image = load_img(img_path, target_size=self.input_size)
        label = load_img(lbl_path, target_size=self.input_size, color_mode="grayscale")

        image = img_to_array(image) / 255.0
        label = img_to_array(label) / 255.0
        label = (label > 0).astype(np.float32) # Convert to binary mask [0, 1]

        return image, label

    def __init__(self, image_dir, label_dir, input_size=(128, 128)):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.input_size = input_size
        self.image_files = sorted([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))])
        self.label_files = sorted([f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))])

        self.image_label_pairs = self.match_image_label_files()

    def match_image_label_files(self):
        pairs = []
        for image_file in self.image_files:
            match = re.match(r'(\w+)\s\((\d+)\)', image_file)
            if match:
                cls, number = match.groups()
                label_file = f'{cls} ({number}).png' # Adjust if label files have a different format
                if label_file in self.label_files:
                    pairs.append((image_file, label_file))
                else:
                    pairs.append((image_file, None))
        return pairs

    def __len__(self):
        return len(self.image_label_pairs)
    
# Datasets and Dataloaders
input_size = (128, 128)
test_dataset = SegmentationDataset(
    image_dir='C:/Users/ehrpr/source/repos/DLIP/Datasets/val/images',
    label_dir='C:/Users/ehrpr/source/repos/DLIP/Datasets/val/masks',
    input_size=input_size
)

# Load the trained model
model = load_model('C:/Users/ehrpr/source/repos/DLIP/Datasets/U_Net_epoch10/unet_final.keras')


# Select random images and labels for visualization
num_images_to_show = 6
random_indices = random.sample(range(len(test_dataset)), num_images_to_show)

# Select random images and labels
images_to_show = [test_dataset[i][0] for i in random_indices]
labels_to_show = [test_dataset[i][1] for i in random_indices]

def visualize_results(images, labels, model):
    fig, axs = plt.subplots(num_images_to_show, 3, figsize=(13, 13))
    
    for i, (image, label) in enumerate(zip(images, labels)):
        # Original image
        axs[i, 0].imshow(image)
        axs[i, 0].axis('off')
        if i == 0:
            axs[i, 0].set_title('Original Image')

        # Ground truth mask
        axs[i, 1].imshow(label[:, :, 0], cmap='gray')
        axs[i, 1].axis('off')
        if i == 0:
            axs[i, 1].set_title('Ground Truth Mask')

        # Prediction
        model_output = model.predict(np.expand_dims(image, axis=0))[0]
        predicted_mask = (model_output > 0.5).astype(np.uint8)

        axs[i, 2].imshow(predicted_mask[:, :, 0], cmap='gray')
        axs[i, 2].axis('off')
        if i == 0:
            axs[i, 2].set_title('Predicted Mask')

    plt.tight_layout()
    plt.show()

# Visualize results
visualize_results(images_to_show, labels_to_show, model)
