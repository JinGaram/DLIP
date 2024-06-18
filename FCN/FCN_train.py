import torch
import torchvision.transforms as transforms
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import re

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
    image_dir='test/images',
    label_dir='test/masks',
    transform=transform,
    label_transform=label_transform
)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Load a Segmentation Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.segmentation.fcn_resnet50(pretrained=True)  # Using FCN ResNet50 model
model.classifier[4] = torch.nn.Conv2d(512, 3, kernel_size=(1, 1), stride=(1, 1))  # Adjust for binary segmentation
model = model.to(device)

# Loss Function and Optimizer
criterion = torch.nn.BCEWithLogitsLoss()  # Binary cross-entropy with logits loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training Function
def train_model(model, dataloader, criterion, optimizer, num_epochs=2):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)['out']
            labels = labels.float()
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Training the Model
num_epochs = 12
train_model(model, train_dataloader, criterion, optimizer, num_epochs)
print("Done!")
