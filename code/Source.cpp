#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "Supp.h"
#include <vector>
#include <string>

using namespace cv;
using namespace cv::ml;
using namespace std;

// Function for preprocessing the input image
Mat preprocessImage(Mat& resizedImage) {

    Mat cannyEdge, normalizedImage, brightenedImage, grayImage, binaryMask, morphKernel, blurredImage;
    Mat morphResult, bilateralFilteredImage;

    // Normalize pixel values to a common scale and normalize RGB channels independently
    resizedImage.convertTo(normalizedImage, CV_32F, 1.0 / 255.0);

    // Apply Gaussian blur to reduce noise and preserve edges
    GaussianBlur(normalizedImage, blurredImage, Size(9, 9), 1.5, 1.5);

    //Convert into grayscale image
    cvtColor(blurredImage, grayImage, COLOR_BGR2GRAY);

    // Increase brightness
    blurredImage.convertTo(brightenedImage, -1, 1.5, 0);

    // Apply Canny edge detection
    Canny(resizedImage, cannyEdge, 150, 200);

    // Define a morphological kernel for further processing
    morphKernel = getStructuringElement(MORPH_RECT, Size(7, 7));

    // Apply morphological closing operation to close gaps in edges
    morphologyEx(cannyEdge, morphResult, MORPH_CLOSE, morphKernel);

    // Apply bilateral filter to preserve edges while reducing noise
    bilateralFilter(morphResult, bilateralFilteredImage, 9, 75, 75);

    return bilateralFilteredImage;
}

int main() {

    // Load the pre-trained SVM model
    Ptr<SVM> svm = Algorithm::load<ml::SVM>("svm_traffic_sign.xml");

    // Define the target size for resizing images
    Size targetSize(256, 256);

    // Define the layout of result windows
    int const noOfImagePerCol = 1, noOfImagePerRow = 2;
    Mat resultWin, win[noOfImagePerRow * noOfImagePerCol], legend[noOfImagePerRow * noOfImagePerCol];

    // Define the directory containing traffic sign images
    string datasetDirectory = "Inputs/Traffic Signs/";

    // Use glob to obtain a list of image file paths in the directory
    vector<string> imagePaths;
    glob(datasetDirectory + "*.png", imagePaths);

    // Loop through each image in the dataset
    for (const string& imagePath : imagePaths) {

        Mat sourceImage = imread(imagePath);

        if (sourceImage.empty()) {
            cout << "Error: Unable to load the image " << imagePath << endl;
            continue; // Skip this image and continue with the next one
        }

        Mat resizedImage, grayImage, mask, result;

        // Resize the image to the target size
        resize(sourceImage, resizedImage, targetSize);
        cvtColor(resizedImage, grayImage, COLOR_BGR2GRAY);

        HOGDescriptor hog;
        hog.winSize = Size(256, 256);
        vector<float> descriptors;

        // Compute HOG descriptors for the preprocessed image
        hog.compute(grayImage, descriptors);

        Mat input(1, static_cast<int>(descriptors.size()), CV_32FC1);

        for (size_t i = 0; i < descriptors.size(); i++) {

            input.at<float>(0, static_cast<int>(i)) = descriptors[i];
        }

        int signClass = static_cast<int>(svm->predict(input));

        if (signClass == 0) { //Circular sign

            Mat				gray;
            Point2i			center;
            vector<Vec3f>	circles;

            cvtColor(sourceImage, gray, COLOR_BGR2GRAY);

            // Apply Hough Circle detection to find circles in the image
            HoughCircles(gray, circles, HOUGH_GRADIENT_ALT, 3, 1, 300, 0.85, 3, -1);

            // find the largest circle
            if (circles.empty())
            {
                continue;
            }

            Vec3i max_circle = circles[0];

            for (size_t i = 1; i < circles.size(); i++)
            {
                Vec3i tmp = circles[i];

                if (max_circle[2] < tmp[2])
                {
                    max_circle = tmp;
                }
            }

            Mat canvasGray(sourceImage.rows, sourceImage.cols, CV_8U);
            canvasGray = 0;
            circle(canvasGray, Point(max_circle[0], max_circle[1]), max_circle[2], Scalar(255), 1, LINE_AA);

            Moments M = moments(canvasGray);
            center.x = M.m10 / M.m00;
            center.y = M.m01 / M.m00;

            // generate mask of the sign
            floodFill(canvasGray, center, 255); // fill inside sign boundary
            cvtColor(canvasGray, canvasGray, COLOR_GRAY2BGR);

            // use the mask to segment the color portion from image
            Mat canvasColor;
            canvasColor = Scalar(0, 0, 0);
            canvasColor = canvasGray & sourceImage;

            resize(canvasColor, result, targetSize);
        }

        else if (signClass == 1) { //Triangular or Octagon sign

            vector<vector<Point>> contours;
            vector<Vec4i> hierarchy;

            grayImage = preprocessImage(sourceImage);

            // Find contours in the preprocessed image
            findContours(grayImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

            size_t longestContourIndex = 0;
            size_t maxContourSize = 0;

            // Find the longest contour (assumed to be a triangle)
            for (size_t i = 0; i < contours.size(); i++) {
                if (contours[i].size() > maxContourSize) {
                    maxContourSize = contours[i].size();
                    longestContourIndex = i;
                }
            }

            Mat triangleMask = Mat::zeros(sourceImage.size(), CV_8U);

            // Draw the longest contour (triangle) on the mask
            drawContours(triangleMask, contours, static_cast<int>(longestContourIndex), Scalar(255), FILLED);

            resize(triangleMask, triangleMask, targetSize);

            // Copy the masked region to the result image
            resizedImage.copyTo(result, triangleMask);
        }

        else {

            cout << "Other kind of shape is detected in image: " << imagePath << endl;
        }

        // Display the original and result images in separate windows
        createWindowPartition(resizedImage, resultWin, win, legend, noOfImagePerCol, noOfImagePerRow);
        resizedImage.copyTo(win[0]);
        putText(legend[0], "Original", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
        result.copyTo(win[1]);
        putText(legend[1], "Result", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

        imshow("Traffic sign segmentation", resultWin);

        waitKey();
    }

    destroyAllWindows();
    return 0;
}
