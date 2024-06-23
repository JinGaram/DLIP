# -------------------------------------------------------------------------------------------------
# * @author  21900727 Garam Jin & 22100034 Eunji Ko
# * @Date    2024-06-24
# * @Mod	 2024-06-10 by YKKIM
# * @brief   Final Project(DLIP)
# -------------------------------------------------------------------------------------------------

import torch
import torchvision.transforms as transforms
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import re
import numpy as np
import random
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import cv2

# Dataset Class
class SegmentationDataset(Dataset):
    def __getitem__(self, idx):
        img_name, lbl_name = self.image_label_pairs[idx]
        img_path = os.path.join(self.image_dir, img_name)
        lbl_path = os.path.join(self.label_dir, lbl_name)

        image = Image.open(img_path).convert("RGB")
        label = Image.open(lbl_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        if self.label_transform:
            label = self.label_transform(label)
            label = (label > 0).float()  # Convert to binary mask [0, 1]

        return image, label, img_name

    def __init__(self, image_dir, label_dir, transform=None, label_transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.label_transform = label_transform
        self.image_files = sorted([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))])
        self.label_files = sorted([f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))])

        self.image_label_pairs = self.match_image_label_files()

    def match_image_label_files(self):
        pairs = []
        for image_file in self.image_files:
            match = re.match(r'(\w+)\s\((\d+)\)', image_file)
            if match:
                cls, number = match.groups()
                label_file = f'{cls} ({number}).png'  # Adjust if label files have a different format
                if label_file in self.label_files:
                    pairs.append((image_file, label_file))
                else:
                    pairs.append((image_file, None))
        return pairs

    def __len__(self):
        return len(self.image_label_pairs)

# Transforms for images and labels
input_size = (224, 224)
transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
label_transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor()
])

# Datasets and Dataloaders
train_dataset = SegmentationDataset(
    image_dir='train/images',
    label_dir='train/masks',
    transform=transform,
    label_transform=label_transform
)
test_dataset = SegmentationDataset(
    image_dir='val/images',
    label_dir='val/masks',
    transform=transform,
    label_transform=label_transform
)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Load the trained model
model_path = "./FCNmodel.pth"
model = torch.load(model_path)
model.eval()

# Select random images and labels
num_images_to_show = 9
random_indices = random.sample(range(len(test_dataset)), num_images_to_show)
images_to_show = [test_dataset[i] for i in random_indices]

def determine_class(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return 1, 0  # normal (no mask detected)
    
    contour = contours[0]
    
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if perimeter == 0:
        return 1, 0  # normal (no valid contour)
    
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    if circularity > 0.8:
        return 2, area  # benign (round mask)
    
    else:
        return 3, area  # malignant (irregular mask)

# Class info
class_info = {
    1: {'color': (0, 1, 0), 'name': 'normal'},      # normal (green)
    2: {'color': (1, 1, 0), 'name': 'benign'},    # benign (yellow)
    3: {'color': (1, 0, 0), 'name': 'malignant'}    # malignant (red)
}

def visualize_results(images, model):
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))

    for i, (image, _, img_name) in enumerate(images):
        row, col = divmod(i, 3)
        # Original image
        original_image = TF.to_pil_image(image)

        # Prediction
        with torch.no_grad():
            model_output = model(image.unsqueeze(0).to(device))['out']
            predicted_mask = torch.sigmoid(model_output)[0][0]  # Assuming binary output
            predicted_mask = (predicted_mask > 0.5).cpu().numpy().astype(np.uint8) * 255

        # Determine class and area
        class_id, area = determine_class(predicted_mask)
        color = class_info[class_id]['color']
        class_name = class_info[class_id]['name']

        # Determine stage for malignant class
        if class_id == 3:
            if area < 100:
                stage = '1'
            elif 100 <= area < 200:
                stage = '2'
            elif 200 <= area < 300:
                stage = '3'
            else:
                stage = '4'
            title = f'Class: {class_name}, Stage: {stage}'
        else:
            title = f'Class: {class_name}'

        # Draw contours on the original image
        predicted_contours, _ = cv2.findContours(predicted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        original_image_with_contours = np.array(original_image)
        cv2.drawContours(original_image_with_contours, predicted_contours, -1, [int(c * 255) for c in color], 2)

        axs[row, col].imshow(original_image_with_contours)
        axs[row, col].axis('off')
        axs[row, col].set_title(img_name, fontsize=10)

        # Display additional information below the image in black color
        axs[row, col].text(0.5, -0.1, title, size=10, ha="center", transform=axs[row, col].transAxes, color='black')

    plt.tight_layout()
    plt.show()

# Convert model to appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Visualize results
visualize_results(images_to_show, model)
