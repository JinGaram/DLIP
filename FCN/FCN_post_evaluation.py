import torch
import torchvision.transforms as transforms
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import re
import numpy as np
import cv2
from sklearn.metrics import precision_score, recall_score, accuracy_score

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
test_dataset = SegmentationDataset(
    image_dir='val/images',
    label_dir='val/masks',
    transform=transform,
    label_transform=label_transform
)

test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Load the trained model
model_path = "./FCNmodel.pth"
model = torch.load(model_path)
model.eval()

# Get all images and labels from the test dataset
images_to_show = [test_dataset[i] for i in range(len(test_dataset))]

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
    1: {'color': (0, 1, 0), 'name': 'normal'},      # (green)
    2: {'color': (1, 1, 0), 'name': 'benign'},      # (yellow)
    3: {'color': (1, 0, 0), 'name': 'malignant'}    # (red)
}

def parse_actual_class_from_filename(filename):
    if "normal" in filename:
        return 1
    elif "benign" in filename:
        return 2
    elif "malignant" in filename:
        return 3
    else:
        return None

def evaluate_results(images, model):
    y_true = []
    y_pred = []

    for image, _, img_name in images:
        # Prediction
        with torch.no_grad():
            model_output = model(image.unsqueeze(0).to(device))['out']
            predicted_mask = torch.sigmoid(model_output)[0][0]  # Assuming binary output
            predicted_mask = (predicted_mask > 0.5).cpu().numpy().astype(np.uint8) * 255

        # Determine class and area
        class_id, _ = determine_class(predicted_mask)

        # Get actual class from filename
        actual_class = parse_actual_class_from_filename(img_name)
        y_true.append(actual_class)
        y_pred.append(class_id)
        
        # Determine correctness
        correctness = "맞음" if actual_class == class_id else "틀림"
        print(f'{img_name}: {correctness}')

    # Calculate precision, recall, and accuracy
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f'(Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Number of Images: {len(images)}')

# Convert model to appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Evaluate results
evaluate_results(images_to_show, model)
