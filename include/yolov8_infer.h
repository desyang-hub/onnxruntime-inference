#pragma once

#include "OnnxInfer.h"

class Yolov8Infer : public OnnxInfer {
private:
    std::vector<std::string> labels_ = {
        "person",        "bicycle",      "car",           "motorcycle",
        "airplane",      "bus",          "train",         "truck",
        "boat",          "traffic light", "fire hydrant",  "stop sign",
        "parking meter", "bench",        "bird",          "cat",
        "dog",           "horse",        "sheep",         "cow",
        "elephant",      "bear",         "zebra",         "giraffe",
        "backpack",      "umbrella",     "handbag",       "tie",
        "suitcase",      "frisbee",      "skis",          "snowboard",
        "sports ball",   "kite",         "baseball bat",  "baseball glove",
        "skateboard",    "surfboard",    "tennis racket",   "bottle",
        "wine glass",    "cup",          "fork",          "knife",
        "spoon",         "bowl",         "banana",        "apple",
        "sandwich",      "orange",       "broccoli",      "carrot",
        "hot dog",       "pizza",        "donut",         "cake",
        "chair",         "couch",        "potted plant",  "bed",
        "dining table",  "toilet",       "tv",            "laptop",
        "mouse",         "remote",       "keyboard",      "cell phone",
        "microwave",     "oven",         "toaster",       "sink",
        "refrigerator",  "book",         "clock",         "vase",
        "scissors",      "teddy bear",   "hair drier",    "toothbrush"
    };

    size_t numAttributes_;
    size_t numPredictions_;
public:
    Yolov8Infer(const ModelConfig& config) : OnnxInfer(config) {
        numAttributes_ = outputShapes_[1];
        numPredictions_ = outputShapes_[2];
    }

    //推理方法
    void run(const cv::Mat& img) override {
        size_t img_width = img.cols;
        size_t img_height = img.rows;

        // 将原始图像填充为正方形，多余部分补充零
        int maxLen = std::max(img_width, img_height);
        cv::Mat paddingImg = cv::Mat::zeros(cv::Size(maxLen, maxLen), CV_8UC3);
        cv::Rect roi(0, 0, img_width, img_height);
        img.copyTo(paddingImg(roi)); // 将图像粘贴到左上角

        // 缩放因子，用于将输出坐标缩放会padding图像
        float x_factor = maxLen / static_cast<float>(inputShapes_[WIDTH]);
        float y_factor = maxLen / static_cast<float>(inputShapes_[HEIGHT]);

        // blobFromImage 将图像转换为4 dim张量
        cv::Mat blobTensor = 
        cv::dnn::blobFromImage(paddingImg, 1.0 / 255.0, 
            cv::Size(inputShapes_[WIDTH], inputShapes_[HEIGHT]), cv::Scalar(), true, false);

        // blobTensor中的像素点个数 b * c * w * h
        size_t pixel_count = inputShapes_[CHANNELS] * inputShapes_[HEIGHT] * inputShapes_[WIDTH];
        // 定义输入张量的形状
        std::vector<int64_t> input_shape_info{1, inputShapes_[CHANNELS], inputShapes_[HEIGHT], inputShapes_[WIDTH]};

        // 准备数据输入
        auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU); // 指定张量存在CPU中

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            allocator_info, 
            blobTensor.ptr<float>(), 
            blobTensor.total(), 
            input_shape_info.data(), 
            input_shape_info.size()
        );


        std::vector<Ort::Value> ort_outputs;
        try
        {
            ort_outputs = session_->Run(
                Ort::RunOptions{},
                inputNames_.data(),
                &input_tensor,
                1,
                outputNames_.data(),
                outputNames_.size()
            );
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            return;
        }

        auto inference_end = std::chrono::high_resolution_clock::now();

        // 解析推理结果
        const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
        cv::Mat det_output0(numAttributes_, numPredictions_, CV_32F, (float*)pdata);

        // 转置，方便后续处理
        cv::Mat det_output;
        cv::transpose(det_output0, det_output);

        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> classIds;

            // 遍历所有检测到的候选框 (det_output的每一行代表一个候选框)
        for (int i = 0; i < det_output.rows; i++) {
            //float confidence = det_output.at<float>(i, 4);
            //if (confidence < 0.45) {
            //	continue;
            //}
            
            // 获得当前目标框的所有类别得分
            cv::Mat classes_scores = det_output.row(i).colRange(4, numAttributes_);

            cv::Point classIdPoint;		// 用于存储分类中的得分最大值索引(坐标)
            double score;				// 用于存储分类中的得分最大值
            minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);
            // 处理分类得分较高的目标框
            if (score > 0.25)
            {
                // 计算在原始图像上,目标框的中心点坐标和宽高
                // 在输入图像上目标框的中心点坐标和宽高
                float cx = det_output.at<float>(i, 0);
                float cy = det_output.at<float>(i, 1);
                float ow = det_output.at<float>(i, 2);
                float oh = det_output.at<float>(i, 3);
                //原始图像上目标框的左上角坐标
                int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
                int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
                //原始图像上目标框的宽高
                int width = static_cast<int>(ow * x_factor);
                int height = static_cast<int>(oh * y_factor);

                // 记录目标框信息
                cv::Rect box;
                box.x = x;
                box.y = y;
                box.width = width;
                box.height = height;

                boxes.push_back(box);
                classIds.push_back(classIdPoint.x);
                confidences.push_back(score);
            }
        }

        // NMS:非极大值抑制，去除同一目标的多余结果。
        std::vector<int> indexes;	// 存储经过 NMS 后保留的目标框在 boxes 向量中的索引
        if (!boxes.empty()) {
            cv::dnn::NMSBoxes(boxes, confidences, 0.25f, 0.45f, indexes);
        }
        std::cout << "get " << indexes.size() << " boxes after NMS." << std::endl;

        

        // 遍历筛选出的目标框
        for (size_t i = 0; i < indexes.size(); i++) {
            try
            {
                int idx = indexes[i];		// 获取当前目标框序号
                int cid = classIds[idx];	// 获取目标框分类得分

                cv::rectangle(img, boxes[idx], cv::Scalar(0, 0, 255), 1, 8, 0);
                cv::putText(img, labels_[cid].c_str(), boxes[idx].br(), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
            }
            catch(const std::exception& e)
            {
                std::cerr << e.what() << '\n';
                return;
            }
        }
        cv::imwrite("./assets/output.jpg", img);
    }
};