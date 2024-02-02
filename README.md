# Traffic-Sign-Segmentation
Traffic sign recognition (TSR) is considered as the outcome of the improving image processing technique and the accelerating process of modernization as well as the increasing number of cars on the road. Previously, the car manufacturers were putting more effort on the driving experience, quality, and comfortability instead of a driver assistance system with better safety aspect. TSR is now a critical component to be included in advanced driver assistance system (ADAS) for autonomous vehicles. Presently, road traffic safety is getting more prioritized since there is significant number of people who lost their lives because of traffic accidents. TSR involves the technique of computer vision (CV) and machine learning to perform object detection, interpretation, and understand the captured traffic signs in real-time.

## Overview
The project aims to develop an advanced traffic sign segmentation system for autonomous vehicles and intelligent transportation systems. It will be designed using the C++ programming language and OpenCV. The system's primary objective is to accurately detect and segment traffic signs from input images, utilizing shape, color, and multistage median filtering techniques to enhance accuracy and reduce false positives. The scope of the project includes image preprocessing for optimized quality, color, and shape analysis for identifying distinctive features, and multistage median filtering to remove noise and undesired blobs.

## Project Objectives
The outlined project aims to create a sophisticated traffic sign segmentation system with the primary goal of accurately detecting and segmenting traffic signs from input images. The system is designed to leverage various approaches, including shape, color, and other techniques, to achieve high segmentation accuracy and robustness. The ultimate objective is to make this system suitable for integration into autonomous vehicles and intelligent transportation systems, contributing to enhanced road safety and efficient traffic management. The following are the main objectives of this project:
1. **Image Preprocessing Techniques**<br>
The project involves implementing various image preprocessing techniques such as color space transformations, Gaussian filtering, and resizing. These techniques are crucial to optimize the quality of input images for subsequent segmentation processes. Proper preprocessing enhances the system's ability to accurately identify and segment traffic signs.
2. **Algorithm Development**<br>
The project focuses on developing algorithms capable of identifying traffic signs based on both geometric features and color characteristics. This approach ensures precise segmentation even in challenging conditions, where factors like lighting and environmental variations may impact traditional segmentation methods. By incorporating both geometric and color-based information, the system aims to improve accuracy and robustness.
3. **Multistage Median Filtering**<br>
To enhance the quality of segmentation results, the project incorporates multistage median filtering. This technique is employed to clean noisy images and eliminate undesired blobs, contributing to more accurate and reliable segmentation outcomes. Noise reduction is critical for the system's performance, especially when dealing with real-world images that may contain artifacts or disturbances.

## Key Components
This section will outline the key components that collectively contribute to the project, the Traffic Sign Segmentation system to achieve the desired performance and expectation which are aligned with the project objectives. This section will cover a few consecutive steps which are essential to ensure the optimal performance of the system and perform segmentation of traffic signs as many as possible. The steps are data collection, data pre-processing, segmentation, model selection and architecture, and model training.
1. **Data Collection**<br>
The dataset that will be used to perform segmentation consists of 70 traffic signs with different characteristics. For instance, there are regulatory traffic signs which indicate 
speed limits and restrictions such as no entry signs, no U-turn signs, and so on. Also, warning signs such as pedestrian crossing signs and slippery road signs which are triangular are included in the dataset as well. The data collection process will also concern for the performance of the system may vary due to the different colors of traffic signs, hence, the blue color traffic signs will be taken into account to validate its performance. To test the robustness of the system, a variety of traffic signs under certain challenging scenarios such as traffic signs with degraded visibility due to variation in lighting, distorted signs with blurry resolution, and signs with complex backgrounds which is a result of urban environments with multiple objects and textures will be selected.
2. **Data Preprocessing**
* **Resizing**
Initially, the dimensions of the input images will be adjusted to a standardized size to ensure all input images are having consistent dimensions, which will ease the segmentation model to perform its operation later. In fact, most of the machine learning models require a fixed input size to maximize efficiency and prevent any possible errors to occur.
* **Normalization**
Normalization is one of the fundamental steps in the pre-processing of the input images. It involves the scaling of the pixel values of the images to a consistent range which is either between 0 and -1 or -1 and 1. Normalizing the images will enhance the performance of the segmentation model in terms of generalization as the model will become less sensitive to the pixel values.
* **Gaussian Blurring**
Gaussian blurring is a common image pre-processing operation that will be used in computer vision techniques which includes traffic sign segmentation. The main objective of this operation is to enhance the quality of images and improve the performance in subsequent image processing tasks.
* **Brightening**
Visibility of the input images is one of the crucial characteristics to ensure the maximization of the performance of the segmentation model. Therefore, brightening the input images is necessitated to improve the quality of the images by adjusting the brightness levels of the images.
* **Grayscale Conversion**
Gray scale conversion is a process that converts the coloured input image to grayscale which is black and white by assigning a single intensity value to each pixel. After performing gray scale conversion, the luminance-based features are more emphasized, and easier for the segmentation model to detect the traffic signs and segment them out in a better performance based on the brightness difference.
* **Canny Edge Detection**
Edges are the key features to identify the boundaries of the traffic signs, therefore, canny edge detection could assist in the identification of the edges by detecting rapid changes in intensity. After localizing the edges, the outlining of the traffic sign regions will be aided by Canny edge detection. In addition to that, noise and artifacts are the constraints in the segmentation of the traffic signs and they are not avoidable in real-world.
* **Morphological Operation**
Morphological operations involved dilation and erosion which could help to enhance the shape of the region to be segmented. Dilation will expand the boundaries of the region to make them well-connected which contributes to a more accurate masking of the region of interest. In contrast, erosion shrinks the boundaries to separate or thin out the region. A fine-tuned balance between dilation and erosion could achieve a shape enhancement.
* **Bilateral Filtering**
Bilateral filtering enhances the robustness of the segmentation model to reduce the impact of occlusion such as the traffic signs in the input images may be occluded by other objects in the background. In simpler words, it enables the segmentation model to discern relevant details with occluded signs and contribute to more accurate segmentation of traffic signs.
3. **Model Selection & Training**<br>
In this system, a **Histogram of Oriented Gradient (HOG)** features and **Support Vector Machine** is used in the process of training and testing the binary image classifier. HOG will used to extract the discriminative features from the training images, and it will construct a training dataset along with the corresponding class label and the SVM will used as the machine learning algorithm for the classification task. In the training phase, the training image will load and resize to 256x256 pixels. Then, it will be used as the input for training the SVM. The term criteria for the SVM training are 100 iterations and it will stop iterating if it reaches the maximum of 100 iterations. Then, the regularization parameter for the SVM training is set to 0.01 which is used to control the trade-off between maximizing the margin between classes and minimizing the classification errors.
4. **Segmentation**<br>
The segmentation task begins by loading the trained model file which will be used to classify the traffic signs into different classes based on their features. The retrieval of input images will be done by iterating through a list of image file paths that obtained using the ‘glob’ function. After preparing all input images, pre-processing steps will be performed by passing the images into the ‘preprocessImage’ function. After pre-processing the images, the Histogram of Oriented Gradients (HOG) descriptors are computed for each pre-processed image and the computed HOG descriptors are stored in a matrix and used as input to the pre-trained SVM classifier. The SVM classifier will predict the class of traffic signs in the image and there will be only two outputs which are **0** and **1**. Class 0 indicates the traffic sign in the image is circular while class 1 indicates triangular or octagonal signs. The identification of the traffic sign class is followed by the segmentation based on the class predicted by the SVM classifier. There is a **difference between the segmentation task for circular signs and triangular or octagonal signs**.

## Why Use Support Vector Machine(SVM) & Histogram of Oriented Gradient(HOG)?
Support Vector Machine (SVM) and Histogram of Oriented Gradients (HOG) are commonly used in combination for various computer vision tasks, including image segmentation. When applied together, they offer several advantages for image segmentation tasks:<br>
1. **Robust Feature Representation**<br>
* HOG provides a robust feature representation by capturing local object shape and structure. The gradient information in different orientations is used to create a feature vector that describes the distribution of gradient intensities in the image. This can be particularly effective for detecting object boundaries and shapes in the context of image segmentation.
2. **Discriminative Power**<br>
SVM is a powerful classification algorithm that excels at distinguishing between different classes. When combined with HOG features, SVM can effectively learn discriminative patterns for separating different regions or objects in an image. This results in a segmentation model that is capable of accurately classifying and segmenting different parts of an image.
3. **Handling Non-linear Relationships**<br>
SVM can handle non-linear relationships between features, which is beneficial when dealing with complex and varied image data. The combination of HOG features and SVM allows the model to capture intricate patterns and variations in the image, making it suitable for image segmentation tasks where objects may exhibit diverse shapes and appearances.

## Getting Started
In the course of developing the Traffic Sign Segmentation System, several software tools were utilized to facilitate the implementation, testing, and evaluation of the project. This section outlines the key software and tools employed throughout the project lifecycle.
### Prerequisites
1. **C++ Compiler**
* The project was primarily implemented using the C++ programming language. The GCC (GNU Compiler Collection) was utilized as the primary compiler for building and executing the C++ codebase.
2. **Integrated Development Environment(IDE)**
* Microsoft Visual Studio 2022 was chosen as the Integrated Development Environment for its rich set of features that aid in code editing, debugging, and project management.
3. **OpenCV (Open-Source Computer Vision Libraries)**
* OpenCV served as the cornerstone of image processing and computer vision tasks in the project. This library provided a robust set of functions for image loading, preprocessing, feature extraction, and more.

## Usage
This section provides guidance on how to use the provided C++ code for image segmentation tasks using the SVM+HOG model. Follow the steps below to run the script and explore the results. To be mentioned that, there are two C++ code files which are respectively model training file and the main code file. If you are interested in training the model and do not mind to spend time on the long-lasting training process, you are suggested to go ahead and give it a try. Otherwise, you can just directly run the main code file by loading the pre-trained model file. Follow the steps below to run the code and explore the results.
### Steps
1. **Download the Code**
* Download the provided source code file and also the pre-trained model file, then save them in your project directory.
2. **Include Necessary Libraries**
* The very first step of the project is to import the requried libraries.
```bash
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "Supp.h"
#include <vector>
#include <string>
```
3. **Model & Dataset Loading**
* You are required to load the pre-trained model xml file as well as the dataset that containing a huge amount of traffic sign images in various situation.
```bash
// Load the pre-trained SVM model
Ptr<SVM> svm = Algorithm::load<ml::SVM>("svm_traffic_sign.xml");

// Define the directory containing traffic sign images
string datasetDirectory = "Inputs/Traffic Signs/";

// Use glob to obtain a list of image file paths in the directory
vector<string> imagePaths;
glob(datasetDirectory + "*.png", imagePaths);

// Loop through each image in the dataset
for (const string& imagePath : imagePaths) {
    ...
}
```
4. **Image Preprocessing**
* All the preprocessing steps are integrated into a single separated function. All the loaded inputs will be passed to the function to undergo image preprocessing for better model performance before being fed to the model.
```bash
// Function for preprocessing the input image
Mat preprocessImage(Mat& resizedImage) {
    ...
    return bilateralFilteredImage;
}
```
5. **Prediction & Segmentation**
* After preprocessing the images, the pre-trained SVM classifier will classify the preprocessed images into binary class which means the there will be only two output for the model, 0 and 1. The images labelled as 0 are the traffic sign in the image is circular while class 1 indicates triangular or octagonal signs. Afterwards, the traffic signs of both classes will undergo different segmentation processes correponding to their shape.
```bash
int signClass = static_cast<int>(svm->predict(input));

if (signClass == 0) { //Circular sign
    ...
    HoughCircles(...)
}

else if (signClass == 1) { //Triangular or Octagon sign
    ...
}
```

## Contributing
We appreciate your interest in contributing to the Time Series Analysis Model project. Whether you are offering feedback, reporting issues, or proposing new features, your contributions are invaluable. Here's how you can get involved:
### How to Contribute
1. **Issue Reporting**
   * If you encounter any issues or unexpected behavior, please open an issue on the project.
   * Provide detailed information about the problem, including steps to reproduce it.
2. **Feature Requests**
   * Share your ideas for enhancements or new features by opening a feature request on GitHub.
   * Clearly articulate the rationale and potential benefits of the proposed feature.
3. **Pull Requests**
   * If you have a fix or an enhancement to contribute, submit a pull request.
   * Ensure your changes align with the project's coding standards and conventions.
   * Include a detailed description of your changes.
  
## License
The Time Series Analysis Model project is open-source and licensed under the [MIT License](LISENCE). By contributing to this project, you agree that your contributions will be licensed under this license. Thank you for considering contributing to our project. Your involvement helps make this project better for everyone. <br><br>
**Have Fun!** 🚀
