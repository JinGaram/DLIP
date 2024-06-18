import os
import random
import cv2 as cv
import numpy as np
from ultralytics import YOLO
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report

# Load YOLO model
model = YOLO('C:/Users/ehrpr/source/repos/DLIP/Datasets/runs/segment/train26_seg_n_30/weights/best.pt')

# Path image
image_dir = 'C:/Users/ehrpr/source/repos/DLIP/Datasets/val/images'

# All images from path
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# initialize
actual_labels = []
predicted_labels = []

# Predict all images
for img_path in image_files:
    # Load images
    src = cv.imread(img_path)

    # Predict
    result = model.predict(source=src)

    # Apply image from results
    r = result[0]
    dst = r.plot()

    # Labels from real ('benign (10).png' -> 'benign')
    actual_label = os.path.basename(img_path).split('(')[0].strip()
    actual_labels.append(actual_label)

    #  Labels from predict 
    if len(r.boxes) > 0:
        predicted_label_idx = int(r.boxes[0].cls.item())
        predicted_label = r.names[predicted_label_idx]
    else:
        predicted_label = 'None'

    # None = normal
    if actual_label == 'normal' and predicted_label == 'None':
        predicted_label = 'normal'
        is_correct = "맞음"
    else:
        is_correct = "맞음" if actual_label == predicted_label else "틀림"
    
    predicted_labels.append(predicted_label)

    # Show result
    print(f'파일명: {os.path.basename(img_path)}, 예측한 값: {predicted_label}, 실제 값: {actual_label}, 틀림여부: {is_correct}')

# Precision, Recall, Accuracy 
precision = precision_score(actual_labels, predicted_labels, average='macro')
recall = recall_score(actual_labels, predicted_labels, average='macro')
accuracy = accuracy_score(actual_labels, predicted_labels)

print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'Accuracy: {accuracy:.2f}')

# Class Evaluation
print("\nClass-wise report:")
print(classification_report(actual_labels, predicted_labels))

# 9 images random
selected_images = random.sample(image_files, 9)

# plt show
fig, axes = plt.subplots(3, 3, figsize=(8, 8))

for i, img_path in enumerate(selected_images):
    # Load images
    src = cv.imread(img_path)

    # Predict
    result = model.predict(source=src)

    # Apply image from results
    r = result[0]
    dst = r.plot()

    # Labels from real ('benign (10).png' -> 'benign')
    actual_label = os.path.basename(img_path).split('(')[0].strip()

    # Labels from predict 
    if len(r.boxes) > 0:
        predicted_label_idx = int(r.boxes[0].cls.item())
        predicted_label = r.names[predicted_label_idx]
    else:
        predicted_label = 'None'
    
    # None = normal
    if actual_label == 'normal' and predicted_label == 'None':
        predicted_label = 'normal'

     # Show result
    ax = axes[i // 3, i % 3]
    ax.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
    ax.axis('off')
    ax.set_title(f'{actual_label} / {predicted_label}')

plt.tight_layout()
plt.show()