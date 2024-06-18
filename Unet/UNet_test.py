import os
import random
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
import cv2

# Load trained model
model = load_model('C:/Users/ehrpr/source/repos/DLIP/Datasets/U_Net_epoch10/unet_final.keras')

# Path images 
test_image_dir = 'C:/Users/ehrpr/source/repos/DLIP/Datasets/val/images'

# # Image load & Preprocess function
def load_and_preprocess_image(image_path, image_size):
    img = load_img(image_path, target_size=image_size)
    img_array = img_to_array(img)
    img_normalized = img_array / 255.0  # Normalize
    return np.expand_dims(img_normalized, axis=0), img_array  # 배치 차원 추가, 원본 배열 반환

# # Analysis mask & determine the class fuction
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

# 클래스에 따른 색상 및 이름 설정
class_info = {
    1: {'color': [0, 255, 0], 'name': 'normal'},      # normal (green)
    2: {'color': [255, 255, 0], 'name': 'benign'},    # benign (yellow)
    3: {'color': [255, 0, 0], 'name': 'malignant'}    # malignant (red)
}

# List iamges
image_files = os.listdir(test_image_dir)
random.shuffle(image_files)
image_files = image_files[:9]  # Random 9 images

# Image show & Size 
plt.figure(figsize=(12, 12))

for i, image_file in enumerate(image_files):
    image_path = os.path.join(test_image_dir, image_file)
    test_image, original_image = load_and_preprocess_image(image_path, (128, 128))

    # Predict
    predicted_mask = model.predict(test_image)
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8)  # Thresholding to binary mask
    predicted_mask = np.squeeze(predicted_mask)  # Remove mask
    original_mask = cv2.resize(predicted_mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    #Analysis & Determine class
    detected_class, mask_area = determine_class(original_mask)

    # Make the image with color
    segmented_image = original_image.copy()
    segmented_image[original_mask == 1] = class_info[detected_class]['color']

    # Malignant class stage
    if detected_class == 3:
        if mask_area < 100:
            stage = '1'
        elif 100 <= mask_area < 200:
            stage = '2'
        elif 200 <= mask_area < 300:
            stage = '3'
        else:
            stage = '4'
        title = f'Segmented Image - Class {detected_class}: {class_info[detected_class]["name"]}, Stage: {stage}'
    else:
        title = f'Segmented Image - Class {detected_class}: {class_info[detected_class]["name"]}'

    # Show images
    plt.subplot(3, 3, i + 1)
    plt.title(title, pad=10)  # Distance between images
    plt.imshow(segmented_image.astype(np.uint8))
    plt.axis('off')

    # 파일명 출력
    plt.text(0.5, -0.1, image_file, ha="center", va="center", transform=plt.gca().transAxes, fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":3})

plt.tight_layout()
plt.subplots_adjust(hspace=0.5, bottom=0.1)  # Make the space between images
plt.show()
