import torch
import torchvision.transforms as transforms
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import re
from sklearn.metrics import precision_score, recall_score, accuracy_score

# Custom Dataset Class
class SegmentationDataset(Dataset):
    def __getitem__(self, idx):
        img_name, lbl_name = self.image_label_pairs[idx]
        img_path = os.path.join(self.image_dir, img_name)
        lbl_path = os.path.join(self.label_dir, lbl_name) if lbl_name else None

        image = Image.open(img_path).convert("RGB")
        label = Image.open(lbl_path).convert("RGB") if lbl_path else None

        if self.transform:
            image = self.transform(image)
        if self.label_transform and label is not None:
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
            else:
                pairs.append((image_file, None))  # Add unmatched files as well
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

test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load the trained model
model_path = "./FCNmodel.pth"
model = torch.load(model_path)
model.eval()

# Device setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Calculate & Output with result
def calculate_metrics(test_dataloader, model, device):
    gt_sizes = []
    pred_sizes = []
    results = []
    total = 0

    with torch.no_grad():
        for images, labels, img_name in test_dataloader:
            images = images.to(device)
            labels = labels.to(device) if labels is not None else None
            
            outputs = model(images)['out']
            predicted_masks = torch.sigmoid(outputs)
            predicted_masks = (predicted_masks > 0.7).float()
            
            for label, predicted_mask in zip(labels, predicted_masks):
                if label is not None:
                    gt_size = label.sum().item()
                else:
                    gt_size = 0
                pred_size = predicted_mask.sum().item()
                
                if gt_size == 0 and pred_size == 0:
                    results.append((img_name[0], pred_size, gt_size, "맞음"))
                    continue
                
                total += 1
                if gt_size * 0.7 <= pred_size <= gt_size * 1.3:
                    results.append((img_name[0], pred_size, gt_size, "맞음"))
                    gt_sizes.append(1)
                    pred_sizes.append(1)
                else:
                    results.append((img_name[0], pred_size, gt_size, "틀림"))
                    gt_sizes.append(1)
                    pred_sizes.append(0)

    precision = precision_score(gt_sizes, pred_sizes)
    recall = recall_score(gt_sizes, pred_sizes)
    accuracy = accuracy_score(gt_sizes, pred_sizes)
    
    return precision, recall, accuracy, results, total

# Calculate
precision, recall, accuracy, results, total = calculate_metrics(test_dataloader, model, device)

for img_name, pred_size, gt_size, result in results:
    print(f"파일명: {img_name}, 예측한 값: {pred_size}, 실제 값: {gt_size}, 틀림여부: {result}")

print(f"Total images compared: {total}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Accuracy: {accuracy:.4f}")
