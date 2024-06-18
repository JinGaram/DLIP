import os
import random
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import cv2
from sklearn.metrics import precision_score, recall_score, accuracy_score

# Trained model load
model = load_model('C:/Users/ehrpr/source/repos/DLIP/Datasets/U_Net_epoch10/unet_final.keras')

# Path images
test_image_dir = 'C:/Users/ehrpr/source/repos/DLIP/Datasets/val/images'

# Image load & Preprocess function
def load_and_preprocess_image(image_path, image_size):
    img = load_img(image_path, target_size=image_size)
    img_array = img_to_array(img)
    img_normalized = img_array / 255.0  # Normalize
    return np.expand_dims(img_normalized, axis=0), img_array  # 배치 차원 추가, 원본 배열 반환

# Analysis mask & determine the class fuction
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
    if circularity > 0.7:
        return 2, area  # benign (round mask)
    
    else:
        return 3, area  # malignant (irregular mask)

# Class info
class_info = {
    1: {'color': [0, 255, 0], 'name': 'normal'},      # normal (green)
    2: {'color': [255, 255, 0], 'name': 'benign'},    # benign (yellow)
    3: {'color': [255, 0, 0], 'name': 'malignant'}    # malignant (red)
}

# Class from filename
def parse_actual_class_from_filename(filename):
    if "normal" in filename:
        return 1
    elif "benign" in filename:
        return 2
    elif "malignant" in filename:
        return 3
    else:
        return None

# List of images
image_files = os.listdir(test_image_dir)

# Evaluate
def evaluate_results(image_files, model, image_size):
    y_true = []
    y_pred = []
    
    for image_file in image_files:
        image_path = os.path.join(test_image_dir, image_file)
        test_image, original_image = load_and_preprocess_image(image_path, image_size)

        # Predict
        predicted_mask = model.predict(test_image)
        predicted_mask = (predicted_mask > 0.5).astype(np.uint8)  # Thresholding to binary mask
        predicted_mask = np.squeeze(predicted_mask)  # Remove mask
        original_mask = cv2.resize(predicted_mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        #Analysis & Determine class
        detected_class, mask_area = determine_class(original_mask)

        # Get actual class from filename
        actual_class = parse_actual_class_from_filename(image_file)
        y_true.append(actual_class)
        y_pred.append(detected_class)
        
        # Determine correctness
        correctness = "맞음" if actual_class == detected_class else "틀림"
        print(f'{image_file}: {correctness}')

    # Calculate precision, recall, and accuracy
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f'정밀도 (Precision): {precision:.4f}')
    print(f'재현율 (Recall): {recall:.4f}')
    print(f'정확도 (Accuracy): {accuracy:.4f}')
    print(f'비교한 이미지 수: {len(image_files)}')

# Evaluate results
evaluate_results(image_files, model, (128, 128))
