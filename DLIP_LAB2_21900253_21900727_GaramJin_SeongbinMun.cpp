/**
**************************************************************************************
* @author  21900253 SeongBin Mun, 21900727 Garam Jin
* @Date    2024-04-26
* @Mod	   2024-4-26 by YKKIM
* @brief   Deep Learning & Image Processing(DLIP): LAB2: Dimension Measurement with 2D camera
*
**************************************************************************************
*/

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

double dist; // Distance from the pinhole to the objec
double Xfocallength = 3269.0; // Focal length in the X direction from calibration
double Yfocallength = 3284.0; // Focal length in the Y direction from calibration
double Href = 345.0; // Actual length of the reference (Acryl)
double Lref = 235.0; // Actual height of the reference (Acryl)
double realL; // Variable to store the actual length of the object (50x50 square)
double realH; // Variable to store the actual height of the object
int ScaleFactor = 4;
const double MIN_CONTOUR_AREA = 20000; // Minimum contour area
int width1 = 50;
int width2 = 100;

// Function for resizing the image to be visible
void showimage(const Mat& image, const string& windowName, double scaleFactor);

// Image processing function
void processImage(const Mat& image, const string& windowName, int width);

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
    processImage(image1, "Contours 1", width1);
    //processImage(image2, "Contours 2");
    processImage(image3, "Contours 3", width2);
    //processImage(image4, "Contours 4");

    //processImage(image, image2);
    waitKey(0);
    destroyAllWindows();

    return 0;
}

// Resize the image
void showimage(const Mat& image, const string& windowName, double scaleFactor) {

    Mat scaledImage;
    resize(image, scaledImage, Size(), 1.0 / scaleFactor, 1.0 / scaleFactor); // The screen size decreases by the scaleFactor amount

    // Display the image
    cv::imshow(windowName, scaledImage);
}

// Implementation of the image processing function
void processImage(const Mat& image, const string& windowName, int width) {

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
    cout << "Actual width of the object(Given) " << width << "mm" << endl;
    cout << "-----------------------------------------------" << endl;

    // Print real value at Window
    string text = "Height: " + to_string(realL) + " mm, Length: " + to_string(realH) + " mm, " + "Width(Given) :" + to_string(width) + " mm";
    Point Org(100, 300);  
    putText(temp, text, Org, FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 0), 6);

    // Display the image
    namedWindow(windowName, WINDOW_NORMAL);
    showimage(temp, windowName, ScaleFactor);
       
}
