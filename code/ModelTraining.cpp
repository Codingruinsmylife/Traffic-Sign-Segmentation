#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace cv::ml;
using namespace std;

int main() {

    string imagesPath = "Dataset/Train/";
    Mat trainingData;
    vector<int> labels;
    HOGDescriptor hog;
    hog.winSize = Size(256, 256);

    for (int label = 0; label < 2; ++label) {

        string folderPath = imagesPath + to_string(label) + "/";
        vector<String> imagePaths;
        glob(folderPath + "*.png", imagePaths);

        cout << "Training the model (" << label << "/1) ..." << endl;

        for (const string& imagePath : imagePaths) {

            Mat image = imread(imagePath);
            resize(image, image, hog.winSize, 0, 0, INTER_LINEAR_EXACT);

            vector<float> descriptors;
            hog.compute(image, descriptors);

            if (!descriptors.empty()) {
                Mat descriptorMat(1, descriptors.size(), CV_32FC1);
                memcpy(descriptorMat.data, descriptors.data(), descriptors.size() * sizeof(float));

                if (trainingData.empty()) {
                    trainingData = descriptorMat;
                }
                else {
                    vconcat(trainingData, descriptorMat, trainingData);
                }

                labels.push_back(label);
            }
        }
    }

    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setC(0.01);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm->train(trainingData, ROW_SAMPLE, labels);
    svm->save("svm_traffic_sign.xml");

    cout << "Training of model is completed" << endl;

    string testImagesPath = "Dataset/Test/";
    vector<String> testImagePaths;
    vector<int> trueLabels;

    // Load test images and true labels
    for (int label = 0; label < 2; ++label) {
        string folderPath = testImagesPath + to_string(label) + "/";
        vector<String> imagePaths;
        glob(folderPath + "*.png", imagePaths);

        for (const string& imagePath : imagePaths) {
            testImagePaths.push_back(imagePath);
            trueLabels.push_back(label);
        }
    }

    // Load the pre-trained SVM model once, outside the loop
    Ptr<SVM> svm = Algorithm::load<ml::SVM>("svm_traffic_sign.xml");

    // Evaluate the model
    vector<int> predictedLabels;

    for (const string& imagePath : testImagePaths) {
        Mat image = imread(imagePath);
        resize(image, image, hog.winSize, 0, 0, INTER_LINEAR_EXACT);

        vector<float> descriptors;
        hog.compute(image, descriptors);

        if (!descriptors.empty()) {
            Mat descriptorMat(1, descriptors.size(), CV_32F);
            memcpy(descriptorMat.data, descriptors.data(), descriptors.size() * sizeof(float));

            int response = static_cast<int>(svm->predict(descriptorMat));
            predictedLabels.push_back(response);
        }
    }

    // Calculate confusion matrix and other metrics
    Mat confusionMatrix = Mat::zeros(2, 2, CV_32S);

    for (size_t i = 0; i < trueLabels.size(); ++i) {
        confusionMatrix.at<int>(trueLabels[i], predictedLabels[i])++;
    }

    // Calculate precision, recall, f1-score, and support
    for (int i = 0; i < 2; ++i) {
        int tp = confusionMatrix.at<int>(i, i);
        int fp = sum(confusionMatrix.col(i))[0] - tp;
        int fn = sum(confusionMatrix.row(i))[0] - tp;
        int tn = sum(confusionMatrix.diag())[0] - tp;

        double precision = static_cast<double>(tp) / (tp + fp);
        double recall = static_cast<double>(tp) / (tp + fn);
        double f1Score = 2 * precision * recall / (precision + recall);
        int support = tp + fn;

        cout << "Class " << i << ":\n";
        cout << "Precision: " << precision << endl;
        cout << "Recall: " << recall << endl;
        cout << "F1-score: " << f1Score << endl;
        cout << "Support: " << support << endl;
    }

    return 0;
}
