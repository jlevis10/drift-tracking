#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <map>


using namespace std;

const float CONFIDENCE_THRESHOLD = 0.8;
const float NMS_THRESHOLD = 0.4;
const vector<int> targetClasses = {0, 2, 5, 7};  // person, car, bus, truck

vector<string> classNames = {
    "person",    // class ID 0
    "car",       // class ID 2
    "bus",       // class ID 5
    "truck"      // class ID 7
};

struct Detection {
    int classId;
    float confidence;
    cv::Rect box;
};

map<int, Detection> trackedBoxes; // Map to track IDs and detections
int nextId = 0; // Counter for unique IDs

bool boxSelected = false; // Flag to check if box is selected
vector<Detection> detectedBoxes; // Store the selected bounding box
Detection selectedBox; // Store the selected bounding box

void mouseCallback(int event, int x, int y, int flags, void* param);
float calculateIoU(Detection& boxA, const Detection& boxB) {
    int intersectionArea = (boxA.box & boxB.box).area(); // Area of intersection
    int unionArea = boxA.box.area() + boxB.box.area() - intersectionArea; // Area of union
    return static_cast<float>(intersectionArea) / unionArea; // IoU
}
vector<Detection> processOutputs(const cv::Mat& frame, const vector<cv::Mat>& outputs, const vector<int>& targetClasses);
void drawBoundingBoxes(cv::Mat& frame, const vector<Detection>& detections, const vector<string>& classNames, const vector<int>& targetClasses);

int main() {

    // Model configuration
    //cv::dnn::Net net = cv::dnn::readNetFromONNX("/Users/jonahlevis/DRIFT/trackerV1/models/yolov5su.onnx");
    cv::dnn::Net net = cv::dnn::readNetFromDarknet("/Users/jonahlevis/Desktop/yolo-test-cpp/src/yolov3.cfg","/Users/jonahlevis/Desktop/yolo-test-cpp/src/yolov3.weights");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    //net.setPreferableTarget(cv::dnn::DNN_TARGET_FP16);

    cv::VideoCapture cap("/Users/jonahlevis/DRIFT/trackerV1/videos/IMG_4220.mp4");

    if (!cap.isOpened()) {
        std::cerr << "Error opening video file" << std::endl;
        return -1;
    }

    cv::Mat frame;
    bool boxSelected = false; // Flag to check if box is selected
    while (cap.read(frame)) {
        cv::setMouseCallback("Frame", mouseCallback, nullptr);

        //cv::imshow("Frame", frame);
        cv::resize(frame, frame, cv::Size(416, 416));  // YOLOv3 typically uses 416x416 input size
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1/255.0, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);
        net.setInput(blob);

        std::vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        // Process the outputs and get the bounding boxes
        detectedBoxes = processOutputs(frame, outputs, targetClasses);

        // Draw the bounding boxes on the frame
        drawBoundingBoxes(frame, detectedBoxes, classNames, targetClasses);

        cv::imshow("Frame", frame);

        // Wait for a key press
        if (cv::waitKey(10) == 27) { // Press ESC to exit
            break;
        }
    }

    return 0;
}
void mouseCallback(int event, int x, int y, int flags, void* param) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        // Check if the click is inside any bounding box
        for (const auto& box : detectedBoxes) {
            if (box.box.contains(cv::Point(x, y))) {
                selectedBox = box; // Set the selected box
                boxSelected = true; // Mark box as selected
                break;
            }
        }
    }
}

float calculateIoU(const Detection& boxA, const Detection& boxB) {
    // Calculate the intersection rectangle
    int xA = std::max(boxA.box.x, boxB.box.x);
    int yA = std::max(boxA.box.y, boxB.box.y);
    int xB = std::min(boxA.box.x + boxA.box.width, boxB.box.x + boxB.box.width);
    int yB = std::min(boxA.box.y + boxA.box.height, boxB.box.y + boxB.box.height);

    // Calculate the area of intersection
    int intersectionWidth = std::max(0, xB - xA);
    int intersectionHeight = std::max(0, yB - yA);
    int intersectionArea = intersectionWidth * intersectionHeight;

    // Calculate the areas of the individual boxes
    int boxAArea = boxA.box.width * boxA.box.height;
    int boxBArea = boxB.box.width * boxB.box.height;

    // Calculate the area of union
    int unionArea = boxAArea + boxBArea - intersectionArea;

    // Return IoU as a float
    return static_cast<float>(intersectionArea) / static_cast<float>(unionArea);
}


// Function to process network outputs and return the bounding boxes after NMS
vector<Detection> _processOutputs(const cv::Mat& frame, const vector<cv::Mat>& outputs, const vector<int>& targetClasses) {
    vector<int> classIds;
    vector<float> confidences;
    vector<cv::Rect> boxes;

    // Iterate through outputs and extract boxes and class information
    for (size_t i = 0; i < outputs.size(); ++i) {
        float* data = (float*)outputs[i].data;
        for (int j = 0; j < outputs[i].rows; ++j, data += outputs[i].cols) {
            float confidence = data[4];
            if (confidence >= CONFIDENCE_THRESHOLD) {
                int classId = max_element(data + 5, data + outputs[i].cols) - (data + 5);
                float score = data[5 + classId];

                // Only process the target traffic-related classes
                if (score >= CONFIDENCE_THRESHOLD && find(targetClasses.begin(), targetClasses.end(), classId) != targetClasses.end()) {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classId);
                    confidences.push_back(score);
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
        }
    }

    // Apply non-maximum suppression (NMS)
    vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);

    // Store the remaining bounding boxes and associated information in a vector of `Detection`
    vector<Detection> detections;
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        detections.push_back({classIds[idx], confidences[idx], boxes[idx]});
    }

    return detections;
}

vector<Detection> processOutputs(const cv::Mat& frame, const vector<cv::Mat>& outputs, const vector<int>& targetClasses) {
    vector<int> classIds;
    vector<float> confidences;
    vector<cv::Rect> boxes;

    for (size_t i = 0; i < outputs.size(); ++i) {
        float* data = (float*)outputs[i].data;
        for (int j = 0; j < outputs[i].rows; ++j, data += outputs[i].cols) {
            float confidence = data[4];
            if (confidence >= CONFIDENCE_THRESHOLD) {
                int classId = max_element(data + 5, data + outputs[i].cols) - (data + 5);
                float score = data[5 + classId];

                if (score >= CONFIDENCE_THRESHOLD && find(targetClasses.begin(), targetClasses.end(), classId) != targetClasses.end()) {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classId);
                    confidences.push_back(score);
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
        }
    }

    vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);

    vector<Detection> detections;
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        detections.push_back({classIds[idx], confidences[idx], boxes[idx]});
    }

    // Match existing boxes or assign new IDs
    std::map<int, Detection> newTrackedBoxes;
    for (const auto& detection : detections) {
        float maxIoU = 0.0;
        int bestMatchId = -1;

        // Find best matching box by IoU
        for (const auto& [id, trackedBox] : trackedBoxes) {
            float iou = calculateIoU(trackedBox, detection);
            if (iou > maxIoU) {
                maxIoU = iou;
                bestMatchId = id;
            }
        }

        if (maxIoU > 0.5) {
            // Update existing box
            newTrackedBoxes[bestMatchId] = detection;
        } else {
            // Assign new ID
            newTrackedBoxes[nextId++] = detection;
        }
    }

    trackedBoxes = std::move(newTrackedBoxes);
    return detections;
}


// Function to draw the bounding boxes on the frame
void _drawBoundingBoxes(cv::Mat& frame, const std::vector<Detection>& detections, 
                       const std::vector<std::string>& classNames, const std::vector<int>& targetClasses) {
    

    for (const auto& detection : detections) {
        const cv::Rect& box = detection.box;
        float iou = calculateIoU(selectedBox, detection);
        // Determine color for the bounding box
        cv::Scalar color = (boxSelected && iou) > 0.5 
                           ? cv::Scalar(255, 0, 0) // Blue for selected boxes
                           : cv::Scalar(0, 255, 0); // Green for others

        
        if (iou > 0.5) {
            selectedBox = detection;
        }
        // Draw the bounding box
        cv::rectangle(frame, box, color, 2);

        // Prepare label for the bounding box
        std::string label = cv::format("%.2f", detection.confidence);
        label = classNames[targetClasses[detection.classId]] + ": " + label;

        // Calculate the position to draw the label
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int top = std::max(box.y, labelSize.height);
        cv::putText(frame, label, cv::Point(box.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

void drawBoundingBoxes(cv::Mat& frame, const std::vector<Detection>& detections, 
                       const std::vector<std::string>& classNames, const std::vector<int>& targetClasses) {
    for (const auto& [id, detection] : trackedBoxes) {
        const cv::Rect& box = detection.box;
        float iou = calculateIoU(selectedBox, detection);

        // Determine color
        cv::Scalar color = (boxSelected && iou > 0.5)
                           ? cv::Scalar(255, 0, 0) // Blue for selected boxes
                           : cv::Scalar(0, 255, 0); // Green for others

        if (iou > 0.5) {
            selectedBox = detection;
        }

        // Draw bounding box
        cv::rectangle(frame, box, color, 2);

        // Prepare label with ID and class name
        std::string label = cv::format("ID: %d | %.2f", id, detection.confidence);
        label = classNames[targetClasses[detection.classId]] + ": " + label;

        // Calculate position for label
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int top = std::max(box.y, labelSize.height);
        cv::putText(frame, label, cv::Point(box.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}
