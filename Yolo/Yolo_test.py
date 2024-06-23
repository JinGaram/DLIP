# -------------------------------------------------------------------------------------------------
# * @author  21900727 Garam Jin & 22100034 Eunji Ko
# * @Date    2024-06-24
# * @Mod	 2024-06-10 by YKKIM
# * @brief   Final Project(DLIP)
# -------------------------------------------------------------------------------------------------

import os
import random
import cv2 as cv
import numpy as np
from ultralytics import YOLO
from matplotlib import pyplot as plt

# YOLO Model
model = YOLO('C:/Users/ehrpr/source/repos/DLIP/Datasets/runs/segment/train26_seg_n_30/weights/best.pt')

# Images path
image_dir = 'C:/Users/ehrpr/source/repos/DLIP/Datasets/val/images'

# Images from list
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# 9 random images
selected_images = random.sample(image_files, 9)

# plt show
fig, axes = plt.subplots(3, 3, figsize=(7, 7))

for i, img_path in enumerate(selected_images):
    # Load images
    src = cv.imread(img_path)

    # Predict
    result = model.predict(source=src, save=True, save_txt=True)

    # Apply results
    r = result[0]
    dst = r.plot()

    # subplot show
    ax = axes[i // 3, i % 3]
    ax.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
    ax.axis('off')
    ax.set_title(os.path.basename(img_path))

plt.tight_layout()
plt.show()
