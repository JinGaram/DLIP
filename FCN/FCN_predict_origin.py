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

# Custom Dataset Class
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
            label = (label > 0).float() # Convert to binary mask [0, 1]

         return image, label 

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
          label_file = f'{cls} ({number}).png' # Adjust if label files have a different format
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

import torch

# Load the trained model
model_path = "./FCNmodel.pth"
model = torch.load(model_path)
model.eval()

import random

# Assuming test_dataset is already defined
num_images_to_show = 9
random_indices = random.sample(range(len(test_dataset)), num_images_to_show)

# Select random images and labels
images_to_show = [test_dataset[i][0] for i in random_indices]
labels_to_show = [test_dataset[i][1] for i in random_indices]

import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

def visualize_results(images, labels, model):
    fig, axs = plt.subplots(num_images_to_show, 3, figsize=(15, 30))

    for i, (image, label) in enumerate(zip(images, labels)):
        # Original image
        axs[i, 0].imshow(TF.to_pil_image(image))
        axs[i, 0].axis('off')
        axs[i, 0].set_title('Original Image')

        # Ground truth mask
        axs[i, 1].imshow(TF.to_pil_image(label))
        axs[i, 1].axis('off')
        axs[i, 1].set_title('Ground Truth Mask')

        # Prediction
        with torch.no_grad():
            model_output = model(image.unsqueeze(0).to(device))['out']
            predicted_mask = torch.sigmoid(model_output)[0][0]  # Assuming binary output
            predicted_mask = (predicted_mask > 0.5).cpu().numpy().astype(np.uint8) * 255

        axs[i, 2].imshow(predicted_mask, cmap='gray')
        axs[i, 2].axis('off')
        axs[i, 2].set_title('Predicted Mask')

    plt.tight_layout()
    plt.show()

# Convert model to appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Visualize results
visualize_results(images_to_show, labels_to_show, model)

