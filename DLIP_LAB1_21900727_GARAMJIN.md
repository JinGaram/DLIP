# LAB: Grayscale Image Segmentation



**Date:**  March-31-2024

**Author:**  GARAM  JIN  21900727

**Github:** https://github.com/JinGaram/DLIP/blob/master/DLIP_LAB1_GARAMJIN

---



# Introduction

## 1. Objective
Recognizing and counting the numbers of the objects that Nuts&Bolts using OpenCV with C++. I am going to utilize the Threshold(Threshold Triangle), Median Filter(Smoothing), and Morphology(Dilation& Erosion).   

**Goal**: Count the number of nuts & bolts of each size for a smart factory automation

There are 2 different size bolts and 3 different types of nuts. You are required to segment the object and count each part of 

* Bolt M5

* Bolt M6

* Square Nut M5

* Hexa Nut M5

* Hexa Nut M6

  

## 2. Preparation

### Software Installation

- OpenCV 4.90,  Visual Studio 2022

### Dataset

The below link is a picture that has some of Bolts and Nuts. In the picture, there are many kinds of Bolts and Nuts. This is a colored picture.

**Dataset link:** [Download the test image](https://github.com/ykkimhgu/DLIP-src/blob/main/LAB_grayscale/Lab_GrayScale_TestImage.jpg)



# Algorithm

## 1. Overview 

The below diagram is composed of 6 steps. Also, I can get the information on the numbers of the nuts and bolts. 

![DLIP_Grayscale1.drawio](C:\Users\ehrpr\source\repos\DLIP\LAB\DLIP_Grayscale1.drawio.png)

​							 	*Fig1. Overview Diagram*

## 2. Procedure

### Loading Grayscale Image & Analysis Histogram

The histogram of the input image is analyzed to understand the distribution of intensity values. It is observed that the bright components of objects can be segmented from mostly dark backgrounds. By analyzing the histogram in Fig.2, it is evident that the image is dark and has low contrast. Therefore, it was decided to set a threshold value using a Thresholding algorithm, either OTSU or Triangle technique. OTSU automatically sets the threshold value between the two peaks, while Triangle determines the threshold value by dropping a perpendicular from the midpoint of the line connecting the two peaks. It was anticipated that the threshold value might be skewed towards one side compared to OTSU. The original source image retains some noise for analysis purposes but aims to preserve the original form. Accordingly, Threshold Triangle was chosen.

![image-20240331124620978](C:\Users\ehrpr\AppData\Roaming\Typora\typora-user-images\image-20240331124620978.png)

​							 *Fig2. Grayscale Image & Histogram*

### Thresholding

![image-20240331124647091](C:\Users\ehrpr\AppData\Roaming\Typora\typora-user-images\image-20240331124647091.png)

​								 *Fig3. Thresholding Image*

### Filtering

A median filter was applied to the input image due to the presence of noticeable salt noise. After performing Thresholding, the resulting image contained a significant amount of noise. Consequently, a Smoothing Filter capable of noise reduction was applied. Smoothing Filters aim to remove noise and small details through blurring. Among them, the non-linear Median Filter was chosen. The reason for this choice is that the Median Filter is effective against impulse noise and prevents excessive smoothing. To compare, a linear filter was also applied, but it did not completely remove the noise and failed to clearly differentiate contours. For these reasons, the author opted for the Median Filter. Fig.4 below shows the image file after applying the Median Filter. Compared to Fig.3, it is evident that noise has been effectively removed, resulting in an overall smoother appearance.

![image-20240331124717776](C:\Users\ehrpr\AppData\Roaming\Typora\typora-user-images\image-20240331124717776.png)

​					     			 *Fig4. Filtering Image*

### Morphology

First, Dilation was applied in Morphology. The purpose of applying a high level of Dilation was to remove holes present in the bolts. However, this caused the holes in the nuts to disappear as well. Subsequently, Erosion was applied to reduce the size again. The purpose of reducing the size was to separate the Rect nut, which was attached. However, when Morphology was performed using this method, the size of both nuts and bolts became significantly smaller than in the original image. To address this issue, a new code was written when drawing contours.

![image-20240331124702288](C:\Users\ehrpr\AppData\Roaming\Typora\typora-user-images\image-20240331124702288.png)

​								 *Fig5. Morphology Image*



# Result and Discussion

## 1. Contour & Final Output

When extracting lines from the Morphology processed image, it was noticed that the sizes did not match when drawing contours on the actual image. Consequently, the size of the rectangle was set to be 1.3 times larger to ensure proper drawing. By using contours, it was possible to measure the size of each object and distinguish them accordingly. Contours were visualized on the image using the cv::rectangle function to draw rectangles.

![image-20240331124823371](C:\Users\ehrpr\AppData\Roaming\Typora\typora-user-images\image-20240331124823371.png)

​								 *Fig6. Contour Image*

The rectangles were drawn with colors corresponding to the objects as follows: M5 Bolts in blue, M6 Bolts in green, M5 Hex Nuts in yellow, M6 Hex Nuts in turquoise, and M5 Rect Nuts in red. Counts were incremented using Count++ to tally the number of each nuts and bolts. Below are the obtained output values.

![image-20240331124838056](C:\Users\ehrpr\AppData\Roaming\Typora\typora-user-images\image-20240331124838056.png)

​							   	 *Fig7. Result Output*

## 2. Discussion

I can see that the actual counts of bolts and nuts match the counts recognized by the code. Below is a summary of the comparison results in Table 1. Recognition was based on comparing the sizes of each bolt and nut, demonstrating the successful implementation of an algorithm capable of separating and recognizing objects.

|    Items    | True | Estimated | Accuracy |
| :---------: | :--: | :-------: | :------: |
|   M5 Bolt   |  5   |     5     |   100%   |
|   M6 Bolt   |  3   |     3     |   100%   |
| M5 Hex Nut  |  4   |     4     |   100%   |
| M6 Hex Nut  |  4   |     4     |   100%   |
| M5 Rect Nut |  5   |     5     |   100%   |

​							   	 *Table1. Result Output*

Since this project objective is to obtain a detection accuracy of 100% for each item, the proposed algorithm has achieved the project goal successfully.



# Conclusion

I have successfully obtained the desired results. We were able to recognize the number, size, and length of each object accurately, and encountered no issues in counting. Moreover, we achieved the goal of this lab by drawing contours for each object. However, there is one regrettable point: we were unable to recognize the inner holes of the nuts. To separate and recognize nuts, we had to apply strong Morphology values, which resulted in the disappearance of the inner holes. Although this issue did not affect the results, there remains a sense of regret for not being able to recognize everything.

---

# Appendix

https://github.com/JinGaram/DLIP/blob/master/DLIP_LAB1_GARAMJIN
