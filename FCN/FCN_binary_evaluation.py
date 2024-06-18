import torch
import torchvision.transforms as transforms
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import re
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score

# Custom Dataset Class
class SegmentationDataset(Dataset):
    def __getitem__(self, idx):
        img_name, lbl_name = self.image_label_pairs[idx]
        img_path = os.path.join(self.image_dir, img_name)
        lbl_path = os.path.join(self.label_dir, lbl_name)

        image = Image.open(img_path).convert("RGB")
        label = Image.open(lbl_path).convert("L")

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
test_dataset = SegmentationDataset(
    image_dir='val/images',
    label_dir='val/masks',
    transform=transform,
    label_transform=label_transform
)

test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load the trained model
model_path = "./FCNmodel.pth"
model = torch.load(model_path)
model.eval()

# Convert model to appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Function to calculate metrics for the entire dataset
def calculate_metrics_and_display(dataloader, model):
    pred_has_white_all = []
    label_has_white_all = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            outputs = model(images)['out']
            preds = torch.sigmoid(outputs) > 0.5

            for j in range(images.size(0)):
                pred = preds[j].cpu().numpy().astype(np.uint8)
                label = labels[j].cpu().numpy().astype(np.uint8)

                # Check for presence of white pixels (255)
                pred_has_white = np.sum(pred) > 0
                label_has_white = np.sum(label) > 0

                pred_has_white_all.append(pred_has_white)
                label_has_white_all.append(label_has_white)

                # Determine if the prediction is correct based on the presence of white pixels
                correct = pred_has_white == label_has_white
                result = "맞음" if correct else "틀림"

                # Print the results
                img_name = test_dataset.image_files[i * images.size(0) + j]
                print(f'파일명: {img_name}, 예측한 값: {pred_has_white}, 실제 값: {label_has_white}, 틀림여부: {result}')

    return np.array(pred_has_white_all), np.array(label_has_white_all)

# Calculate metrics for the test dataset and display results
all_preds, all_labels = calculate_metrics_and_display(test_dataloader, model)

# Calculate Precision, Recall, and Accuracy
precision = precision_score(all_labels, all_preds, pos_label=True)
recall = recall_score(all_labels, all_preds, pos_label=True)
accuracy = accuracy_score(all_labels, all_preds)

print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'Accuracy: {accuracy:.2f}')
