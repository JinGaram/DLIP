# LAB: Dimension Measurement with 2D camera



**Date:**  April-26-2024

**Author:**  SeongbinMun 21900253, GaramJin 21900727

**Github:** https://github.com/JinGaram/DLIP/blob/master/DLIP_LAB2_21900253_21900727_GaramJin_SeongbinMun.cpp

---



# Introduction

## 1. Objective

I need to measure the complete dimensions of rectangular objects using only a smartphone camera. It should be simple and possible to take a photo and measure it immediately.

**Goal**: A company has tasked me with measuring the sizes of two different rectangular solids.

The given values for the two objects are as follows: an Acryl has been set as the reference object. The specifications of this objects are as follows:

* Block 1: [100 X Unknown X Unknown]

* Block 2: [50 X Unknown X Unknown]

* Acryl: [235 X 345]

  ※Units: [mm], Dimensions: [Width X Length X Height]
  
  

## 2. Preparation

### Software Installation

- OpenCV 4.90,  Visual Studio 2022

### Dataset

The below link is the pictures that have Blocks and Acryl. These are the colored pictures.

**Dataset link:** 

[The Test Image 1]: https://github.com/JinGaram/DLIP/blob/master/Ref%20%26%20Object_1.jpg
[The Test Image 2]: https://github.com/JinGaram/DLIP/blob/master/Ref%20%26%20Object_3.jpg


# Algorithm

## 1. Overview 

We made a diagram using five steps and ultimately determined the object's size.

![image-20240426000250473](C:\Users\ehrpr\AppData\Roaming\Typora\typora-user-images\image-20240426000250473.png)

​							 	*Fig1. Overview Diagram*



## 2. Procedure

### Introduction & Calibration(Focal Length)

Our group decided to use the Pinhole Camera Model to measure distances. We introduced three assumptions for this, but after finding that the third assumption caused significant errors, we removed it. Instead of the third assumption, we changed our measurement method. The assumptions are as follows.

​	![가정](https://github.com/sbMunhgu/Picture/assets/145007994/920f1602-9a45-46ae-92a3-2ccfdea5ff16)

​						 	      	*Fig2. Assumption*s

The Pinhole Camera Model can be seen in the following diagram.

​	![pinhole](https://github.com/sbMunhgu/Picture/assets/145007994/8b0b7539-dbaa-4755-8172-c86fe7aa7bc8)

​					      		 *Fig3. Pinhole Camera model*

From this Figure, we can derive the proportional relationship (d:f = H/2:h/2). The focal length (f) can be determined through the calibration below.

![Calibration](https://github.com/sbMunhgu/Picture/assets/145007994/82ef72d0-8986-46bf-a0fa-1c2888d6d398)

​						       		 *Fig4. Calibration*

The focal length of the smartphone used is 3,269 [px], and this is used as a fixed value. The length of H is the known value of the Reference and Block. To determine the length of h, we only need to know the value of h [px]. We carried out the following procedure to calculate this.



### Load the Images 

As mentioned above, instead of using assumption 3, we measured as follows.

![Delete_Assumption](C:\Users\ehrpr\source\repos\DLIP\LAB\LAB2\최종\Image\Delete_Assumption.png)

​					   		     *Fig5. Measurement Method*

We did this to eliminate the distance difference between the Acryl and the Block, which helped remove significant errors in the measurements. Using this method, we took three photos, which include the Reference Acryl, Block 1, and Block 2.

![Load_Image](C:\Users\ehrpr\source\repos\DLIP\LAB\LAB2\최종\Image\Load_Image.png)

​			    			  *Fig6. Load the Images(Block2&Block1)*



### Measuring the Pixel Size of An Image

The pixel values of the length and height of a rectangular object

![image-20240426003325700](C:\Users\ehrpr\AppData\Roaming\Typora\typora-user-images\image-20240426003325700.png)

The pixel values of the length and height of a cube object

![image-20240426003343401](C:\Users\ehrpr\AppData\Roaming\Typora\typora-user-images\image-20240426003343401.png)

I could derive the pixel values for each object using a contour. I can determine the actual distance by substituting these pixel values into the equation below.



### Proportional Expression(Curve Fitting)

After calculating the image's pixel size h[px], the following proportional relationship can be used.
$$
d:f = H/2:h/2
$$
Using OpenCV, we obtained the pixel values for the Reference Acryl, Block 1, and Block 2. 
$$
d_1:3,269[px] = 235/2[mm]:306/2[px]
\\ d_2:3,269[px] = 100/2[mm]:72/2[px]
\\ d_3:3,269[px] = 50/2[mm]:82/2[px]
$$
The value of d came out as follows.
$$
d_1 = 3,557[mm] \ \ \ H_1 = 235[mm] \ \ \ h_1 = 306[px]
\\ d_2 = 4,540[mm] \ \ \ H_1 = 100[mm] \ \ \ h_1 = 72[px]
\\ d_3 = 1,993[mm] \ \ \ H_1 = 50[mm] \ \ \ h_1 = 82[px]
$$
According to *Fig.3*, the value of θ can be derived using the tangent function.
$$
tanθ_1/2=117.5/3,557 \ \ \ θ_1=3.78°
\\ tanθ_2/2=50/4,540 \ \ \ θ_2=1.26°
\\ tanθ_3/2=25/1,993 \ \ \ θ_3=1.42°
$$
Finally, by performing curve fitting on the H/d and θ plane, the shape of the graph can be derived as follows.

<img src="https://github.com/sbMunhgu/Picture/assets/145007994/223d7453-7471-449d-951b-269e656ff182" alt="Curve Fitting" style="zoom:80%;" />

​					       			 *Fig7. Curve Fitting*

Although there are few equations, we can make predictions as follows based on the available data.<img src="https://github.com/sbMunhgu/Picture/assets/145007994/372dd796-fd5b-4c14-9337-30c6534f5a41" alt="그래프 추측" style="zoom:50%;" />

​					        		*Fig8. Estimated Curve Fitting*

We can expect the curve to form similarly to the shape of the tangent function. This is because in *Fig.3*, as the angle θ increases, the value of d decreases, and the value of H increases.



# Result and Discussion

## 1. Derive the Final Image Size

 #### 	1-1. Final Image size 

The final output is as follows.

![image-20240426003035650](C:\Users\ehrpr\AppData\Roaming\Typora\typora-user-images\image-20240426003035650.png)

​									*Fig9. Final Output*

The output was obtained using two photographs, and the analysis of the output values will be continued later.



## 2. Discussion

Our experimental goal is to achieve an error range within 3[mm]. This translates to more than 94% accuracy based on a 50[mm] standard, and more than 97% based on a 100[mm] standard. The results of our experiment are shown in the table below.

​							          	*Table1. Accuracy*

|     Item      |  True   | Estimated  | Accuracy |
| :-----------: | :-----: | :--------: | :------: |
| Block1 Length | 50[mm]  | 51.875[mm] |  96.3%   |
| Block1 Height | 50[mm]  | 49.448[mm] |  98.9%   |
| Block2 Length | 100[mm] | 98.878[mm] |  98.9%   |
| Block2 Height | 50[mm]  | 49.784[mm] |  99.6%   |

We were able to derive successful results within the predicted range. We used the Pinhole Model mentioned above to obtain the results. Initially, when we assumed that height was not considered, we confirmed that the errors were severe. Consequently, we attempted to correct the errors by substituting the given Width into the proportionality equation based on the Pinhole Model. Objects given a width of 50[mm] were successfully corrected, but those given 100[mm] showed less significant correction. This means that the larger the object's width, the more severe the error became. Furthermore, we found that to obtain accurate length and height measurements using the Pinhole Model, the plane of the reference object and the plane of the object being measured must approximately lie on the same plane. Consequently, as shown in Fig5, we decided to place the reference object on top of the object and replaced the reference object with a transparent acrylic plate for the measurement. The results confirmed that the measurements were within the error range.



# Conclusion

I successfully achieved the desired results and recognized the length of each object within the error range. However, one disappointing aspect was that it was impossible to measure with just one photograph. This is because I assumed that the objects on the same plane would be photographed in a parallel position. This entails the inconvenience of having to take photos of each surface you want to measure. Although there were only two objects to measure in this experiment, I anticipate difficulties when photographing many objects in the field. If I had chosen a transparent 3D object instead of a 2D one for the reference setting, I speculate that it might have been possible to measure everything at once.



---

# Appendix

**Code**

```cpp
int main() {
    // Load the image
    Mat image1 = imread("Ref & Object_1.jpg");
    if (image1.empty()) {
        cout << "Failed to load the image." << endl;
        return -1;
    }

    /*Mat image2 = imread("Ref & Object_2.jpg");
    if (image2.empty()) {
        cout << "Failed to load the image." << endl;
        return -1;
    }*/

    Mat image3 = imread("Ref & Object_3.jpg");
    if (image3.empty()) {
        cout << "Failed to load the image." << endl;
        return -1;
    }

    /*Mat image4 = imread("Ref & Object_4.jpg");
    if (image4.empty()) {
        cout << "Failed to load the image." << endl;
        return -1;
    }*/

    // Call the image processing function
    processImage(image1, "Contours 1");
    //processImage(image2, "Contours 2");
    processImage(image3, "Contours 3");
    //processImage(image4, "Contours 4");

    //processImage(image, image2);
    waitKey(0);
    destroyAllWindows();

    return 0;
}
```

-> This is main function.



```cpp
// Implementation of the image processing function
void processImage(const Mat& image, const string& windowName) {

    // Convert to grayscale
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // Apply median filter
    Mat enhanced_image;
    medianBlur(gray, enhanced_image, 9);

    // Detect edges using Canny edge detection
    Mat edges;
    Canny(enhanced_image, edges, 50, 200);

    // Apply morphological operations
    Mat morph_image;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(11, 11));
    morphologyEx(edges, morph_image, MORPH_CLOSE, kernel);

    // Find contours
    vector<vector<Point>> contours;
    findContours(morph_image, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

    // Initialize variables
    double max_area = 0;
    double max_width = 0, max_height = 0;
    double prev_width = 0, prev_height = 0;
    double refL1 = 0, refH1 = 0;
    double ObjectL2 = 0, ObjectH2 = 0;

     Mat temp = image.clone();

      for (const auto& contour : contours) {
          double area = contourArea(contour);

          //For the removal of unnecessary contours, a minimum contour area is set
          if (area > MIN_CONTOUR_AREA) {
              Rect boundingBox = boundingRect(contour);
              rectangle(temp, boundingBox, Scalar(0, 255, 0), 4);
              //Save width & height value
              double width = boundingBox.width;
              double height = boundingBox.height;

              // If the current contour has the largest area
              if (area > max_area) {

                  // Update the current values to the maximum values
                  max_area = area;
                  max_width = width;
                  max_height = height;
                  // Save the pixel values of the maximum length and height
                  refL1 = max_width;
                  refH1 = max_height;
                  // Save the current values as previous values
                  prev_width = max_width;
                  prev_height = max_height;

              }// If the width or height differs from the previous value by more than 800
              else if (abs(width - prev_width) > 800 || abs(height - prev_height) > 800) {
                  // Save the pixel values (internal length and height value)
                  ObjectL2 = width;
                  ObjectH2 = height;
                  break;
              }
              else { // If the difference from the previous value is not more than 800, save the previous value
                  prev_width = width;
                  prev_height = height;
              }
          }
      }

    //print object pixel value
    //cout << "-----------------------------------------------" << endl;
    //cout << "물체의 Pixel 길이값: " << ObjectL2 << "Pixel" << endl;
    //cout << "물체의 Pixel 높이값: " << ObjectH2 << "Pixel" << endl;
    //cout << "-----------------------------------------------" << endl;

    dist = (Xfocallength * Lref) / refL1;  // Proportional equation->d:f=H(actual length of reference):h(pixel length of reference)
    realL = (dist * ObjectL2) / Xfocallength; // Proportional equation->d:f=H(actual length of object):h(pixel length of object)

    dist = (Yfocallength * Href) / refH1; // Proportional equation->d:f=H(actual height of reference):h(pixel height of reference)
    realH = (dist * ObjectH2) / Yfocallength; // Proportional equation->d:f=H(actual height of object):h(pixel height of object)

    // Print real value
    cout << "-----------------------------------------------" << endl;
    cout << "Actual height of the object " << realL << "mm" << endl;
    cout << "Actual length of the object " << realH << "mm" << endl;
    cout << "-----------------------------------------------" << endl;

    // Print real value at Window
    string text = "Height: " + to_string(realL) + " mm, Length: " + to_string(realH) + " mm";
    Point Org(100, 300);  
    putText(temp, text, Org, FONT_HERSHEY_SIMPLEX, 3, Scalar(0, 0, 0), 6);

    // Display the image
    namedWindow(windowName, WINDOW_NORMAL);
    showimage(temp, windowName, ScaleFactor);
       
}
```

-> This is an image processing function, with logic implemented for measuring actual lengths.



```cpp
// Resize the image
void showimage(const Mat& image, const string& windowName, double scaleFactor) {

    Mat scaledImage;
    resize(image, scaledImage, Size(), 1.0 / scaleFactor, 1.0 / scaleFactor); // The screen size decreases by the scaleFactor amount

    // Display the image
    cv::imshow(windowName, scaledImage);
}
```

 ->It uses the scaleFactor parameter to display the screen at a reduced size.
