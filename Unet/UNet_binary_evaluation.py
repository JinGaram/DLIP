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
from sklearn.metrics import precision_score, recall_score, accuracy_score

# Custom Dataset Class
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

# Function to calculate metrics for the entire dataset
def calculate_metrics_and_display(dataset, model):
    pred_has_white_all = []
    label_has_white_all = []

    for i in range(len(dataset)):
        image, label = dataset[i]

        # Predict
        model_output = model.predict(np.expand_dims(image, axis=0))[0]
        predicted_mask = (model_output > 0.5).astype(np.uint8)

        # Check for presence of white pixels (255)
        pred_has_white = np.sum(predicted_mask) > 0
        label_has_white = np.sum(label) > 0

        pred_has_white_all.append(pred_has_white)
        label_has_white_all.append(label_has_white)

        # Determine if the prediction is correct based on the presence of white pixels
        correct = pred_has_white == label_has_white
        result = "맞음" if correct else "틀림"

        # Print the results
        img_name = dataset.image_files[i]
        print(f'파일명: {img_name}, 예측한 값: {pred_has_white}, 실제 값: {label_has_white}, 틀림여부: {result}')

    return np.array(pred_has_white_all), np.array(label_has_white_all)

# Calculate metrics for the test dataset and display results
all_preds, all_labels = calculate_metrics_and_display(test_dataset, model)

# Calculate Precision, Recall, and Accuracy
precision = precision_score(all_labels, all_preds, pos_label=True)
recall = recall_score(all_labels, all_preds, pos_label=True)
accuracy = accuracy_score(all_labels, all_preds)

print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'Accuracy: {accuracy:.2f}')
