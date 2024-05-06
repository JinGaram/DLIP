import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

img1 = cv.imread('LV1.png')
#img1 = cv.imread('LV2.png')
#img1 = cv.imread('LV3.png')
img11 = img1.copy()
img111 = img1.copy()

x=715; w=1200; y=1; h=1075
roi = img1[y:y+h, x:x+w]
x1=0; w1=820; y1=1; h1=388
roi1 = img1[y1:y1+h1, x1:x1+w1]
img1[y1:y1+h1, x1:x1+w1] = [0, 0, 0]
img1[y:y+h, x:x+w] = [0, 0, 0]
# cv.namedWindow('Image with Black ROI', cv.WINDOW_NORMAL)
# cv.imshow('Image with Black ROI', img1)

# Inrange
hsv = cv.cvtColor(img1, cv.COLOR_BGR2HSV)
lower_range = np.array([170, 150, 95])
upper_range = np.array([179, 255, 255])
mask = cv.inRange(hsv, lower_range, upper_range)
dst = cv.bitwise_and(hsv, hsv, mask= mask)
# cv.namedWindow("Filtered Image", cv.WINDOW_NORMAL)
# cv.imshow('Filtered Image', dst)

# Grayscale
gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
# cv.namedWindow('gray', cv.WINDOW_NORMAL)
# cv.imshow('gray', gray)

# Morphology
kernel1 = np.ones((7, 7),np.uint8)
kernel2 = np.ones((55,55),np.uint8)
dilation1 = cv.dilate(gray,kernel1,iterations = 1)
erosion = cv.erode(dilation1,kernel2,iterations = 1)
dilation = cv.dilate(erosion,kernel2,iterations = 1)
# cv.namedWindow('dilation', cv.WINDOW_NORMAL)
# cv.imshow('dilation', dilation)

# Median
kernel = 81
median = cv.medianBlur(dilation,kernel)
# cv.namedWindow('median1', cv.WINDOW_NORMAL)
# cv.imshow('median1', median)

# Contour
contours, hierarchy = cv.findContours(median.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
x2, y2, w2, h2 = cv.boundingRect(contours[0])

#Finding Contour Coordinate and limit the contour part
repoints = []
for contour in contours:
    for point in contour:
        x, y = point.ravel()  
        if x >= x2 and y >= y2 + h2 - 350 // 2 and x <= x2+w2 and y <= y2+h2: #minx, miny, maxx, maxy
            repoints.append((x, y))
repoints1 = [np.array(repoints, dtype=np.int32)]
#print(filtered_points1)

cv.polylines(img11, repoints1, False, (0, 255, 0), 5)
# cv.namedWindow('polylines', cv.WINDOW_NORMAL)
# cv.imshow('polylines', img11)

#Extend the line and Curve fitting
data_x = repoints1[0][:, 0]
data_y = repoints1[0][:, 1]

coefficients = np.polyfit(data_x, data_y, 2)
polynomial = np.poly1d(coefficients)

repoints2 = []
x_line = np.linspace(min(data_x-300), max(data_x+500)).astype(int)
y_line = polynomial(x_line).astype(int)
repoints2.append((x_line, y_line))
repoints3 = [np.array(repoints2, dtype=np.int32)]
#print(repoints3)

for i in range(1, len(x_line)):
    cv.line(img11, (x_line[i-1], y_line[i-1]), (x_line[i], y_line[i]), (0, 255, 0), 5)

#Detect the Conditions
score = img11.shape[0] - max(data_y).astype(float)
# print(max(data_y))
# print("Score :", score)
if max(data_y) < img11.shape[0]-250: Level = 1
elif max(data_y) > img11.shape[0]-250 and max(data_y) < img11.shape[0]-120: Level = 2   
elif max(data_y) > img11.shape[0]-120: Level = 3
# print("Level :", Level)

#Text on Output image
cv.putText(img11, f"Score : {score}", (980, 310), cv.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2, cv.LINE_AA)
cv.putText(img11, f"Level : {Level}", (980, 360), cv.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2, cv.LINE_AA)
cv.rectangle(img11, (950, 250), (1350, 390), (0, 255, 0), 2)

#Line on Output image
for j in range (0, img11.shape[1], 60):
    cv.line(img11, (j, 1002), (j+30, 1002), (255, 200, 0), 3, cv.LINE_4)

for k in range (0, img11.shape[1], 60):
    cv.line(img11, (k, 780), (k+30, 780), (0, 255, 0), 3, cv.LINE_4)

mask1 = cv.drawContours(median, contours, -1, (255, 255, 255), -1)

# cv.namedWindow("mask1", cv.WINDOW_NORMAL)
# cv.imshow('mask1', mask1)
inverse_mask = cv.bitwise_not(mask1)

result = cv.bitwise_and(img11, img11, mask=inverse_mask) #mask1
result1 = cv.bitwise_and(img111, img111, mask=mask1)
Final_Output = cv.add(result, result1)
# cv.namedWindow('Result', cv.WINDOW_NORMAL)
# cv.imshow('Result', result)
# cv.namedWindow('Result1', cv.WINDOW_NORMAL)
# cv.imshow('Result1', result1)
# cv.namedWindow('curve fitting with line', cv.WINDOW_NORMAL)
# cv.imshow('curve fitting with line', img11)
cv.namedWindow('Final_Output', cv.WINDOW_NORMAL)
cv.imshow('Final_Output', Final_Output)

cv.waitKey(0)
