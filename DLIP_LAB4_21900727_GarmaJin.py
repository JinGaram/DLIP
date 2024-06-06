# -------------------------------------------------------------------------------------------------
# * @author  21900727 Garam Jin
# * @Date    2024-06-06
# * @Mod	 2024-04-26 by YKKIM
# * @brief   Deep Learning & Image Processing(DLIP): [LAB4] CNN Object Detection 1
# -------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2 as cv
import numpy as np

#model = YOLO('C:/Users/ehrpr/source/repos/DLIP/LAB/LAB4/runs/detect/train8/weights/best.pt')
model = YOLO('yolov8s')

#Class[2] = car, accuracy of object detected
classes = [2]
conf_thresh = 0.9
iou_thresh = 0.5
video = 'DLIP_parking_test_video.avi'
results = model.predict(source = video, stream = True, save = True, classes = classes, conf = conf_thresh, iou=iou_thresh)
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('21900727.mp4', fourcc, 25, (1280, 720))

#ROI Point
points = np.array([[0, 200], [1280, 200], [1280, 400], [0, 400]], np.int32)
points = points.reshape((-1, 1, 2))

#Parking Point
pts = [
    np.array([[0, 410], [80,325], [177,325], [82,435], [0, 435]]),
    np.array([[183,325], [272,325], [197,435], [88,435]]),
    np.array([[278,325], [363,325], [307,435], [203,435]]),
    np.array([[368,325], [453,325], [417,435], [313,435]]),
    np.array([[458,325], [543,325], [522,435], [423,435]]),
    np.array([[548,325], [633,325], [632,435], [528,435]]),
    np.array([[638,325], [723,325], [742,435], [638,435]]),
    np.array([[728,325], [813,325], [847,435], [748,435]]),
    np.array([[818,325], [903,325], [957,435], [853,435]]),
    np.array([[908,325], [987,325], [1062,435], [963,435]]),
    np.array([[993,325], [1077,325], [1172,435], [1068,435]]),
    np.array([[1083,325], [1172,325], [1277,435], [1178,435]]),
    np.array([[1283,435], [1178,325], [1265,325], [1280,340]]),
]

#Parking center point
parking_centers = [np.mean(pt, axis=0).astype(int) for pt in pts]

# Open the video camera no.0
cap = cv.VideoCapture(video)

if not cap.isOpened():
    print('Cannot open Video')

with open('counting_result.txt', 'w') as f:
    # If not success, exit the program
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        #ROI what I want to detect 
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv.fillPoly(mask, [points], 255)
        roi = cv.bitwise_and(frame, frame, mask=mask)
        
        #Mask for combine
        inverse_mask = cv.bitwise_not(mask)
        
        #Applied roi with yolov8
        results = model(roi)
    
        detected_objects = len(results[0].boxes)
        frame_number = int(cap.get(cv.CAP_PROP_POS_FRAMES))-1
        f.write(f"{frame_number},{detected_objects}\n")
        
        #Show the car class
        plotted = results[0].plot() 
        
        #Combine with roi and non-roi
        befadd_output = cv.bitwise_and(frame, frame, mask=inverse_mask)
        befadd1_output = cv.bitwise_and(plotted, plotted, mask=mask)
        added_output = cv.add(befadd1_output, befadd_output)
    
        #Parking area initial color is green
        colors = [(0, 255, 0)] * 13 #Green
         
        #Detecting box coordinate
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                median_x = (x1+x2)/2
                median_y = (y1+y2)/2 

                #Determining if a point exists inside. If it is right, Red
                for i, center in enumerate(parking_centers):
                    if cv.pointPolygonTest(pts[i], (median_x, median_y), False) >=0:
                        colors[i] = (0, 0, 255) #Red
                        red_count = colors.count((0, 0, 255))
                        green_count = 13 - red_count

                #Draw the polyline each block
                for i in range(13):
                    cv.polylines(added_output, [pts[i]], isClosed=(i != 0 and i != 12), color=colors[i], thickness=2) #Red
                
                #For output
                text = f'Number of Vehicle: {detected_objects}, Available Parking Spaces: {green_count}'

        #Final output
        cv.putText(added_output, text, (30, 30), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv.LINE_AA)
        cv.imshow('YOLOv8 Object Detection', added_output)
        print(f'{text}')
        out.write(added_output)
        
        if cv.waitKey(30) & 0xFF == 27:
            break

cap.release()
cv.destroyAllWindows()