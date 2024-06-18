import os
import cv2 as cv
import numpy as np
from ultralytics import YOLO
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report

# Custom Dataset Class
class ImageDataset:
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = cv.imread(img_path)
        mask_path = self.mask_files[idx]
        if mask_path and os.path.exists(mask_path):
            mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros_like(image[:, :, 0])
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

def calculate_contour_area(mask):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return sum(cv.contourArea(c) for c in contours)

# Image and mask directory paths
image_dir = 'C:/Users/ehrpr/source/repos/DLIP/Datasets/val/images'
mask_dir = 'C:/Users/ehrpr/source/repos/DLIP/Datasets/val/masks_mask'

# Datasets and Dataloaders
image_dataset = ImageDataset(image_dir=image_dir, mask_dir=mask_dir)

# YOLO model load
model = YOLO('C:/Users/ehrpr/source/repos/DLIP/Datasets/runs/segment/train26_seg_n_30/weights/best.pt')

# Initialize
comparison_results = []

# Predict all images
for idx in range(len(image_dataset)):
    img_path, image, actual_mask = image_dataset[idx]

    # Calculate area
    actual_area = calculate_contour_area(actual_mask)

    # Predict
    result = model.predict(source=image, save=False)
    r = result[0]

    # Predict mask
    pred_mask = np.zeros_like(actual_mask)
    for box in r.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        pred_mask[y1:y2, x1:x2] = 255

    predicted_area = calculate_contour_area(pred_mask)

    # Contour size check
    if actual_area == 0 and predicted_area == 0:
        is_correct = "맞음"
    elif actual_area > 0 and predicted_area > 0:
        similarity = predicted_area / actual_area
        is_correct = "맞음" if 0.7 <= similarity <= 1.3 else "틀림"
    else:
        is_correct = "틀림"
    
    comparison_results.append(is_correct)

    # Show result
    print(f'파일명: {os.path.basename(img_path)}, 예측한 컨투어 영역: {predicted_area}, 실제 컨투어 영역: {actual_area}, 틀림여부: {is_correct}')

# '맞음' = 1, '틀림' = 0 convert
binary_actual = [1 if result == '맞음' else 0 for result in comparison_results]
binary_predicted = [1] * len(binary_actual)  # Predict is 맞음

# Precision, Recall, Accuracy 
precision = precision_score(binary_actual, binary_predicted)
recall = recall_score(binary_actual, binary_predicted)
accuracy = accuracy_score(binary_actual, binary_predicted)

print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'Accuracy: {accuracy:.2f}')

# Class Evaluation
print("\nClass-wise report:")
print(classification_report(binary_actual, binary_predicted))
