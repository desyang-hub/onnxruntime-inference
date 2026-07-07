/**
 * @FilePath     : /onnxruntime-infer/examples/yolo_infer/yolo_infer.cpp
 * @Description  :  
 * @Author       : desyang
 * @Date         : 2026-07-01 21:12:45
 * @LastEditors  : desyang
 * @LastEditTime : 2026-07-03 11:02:42
**/
#include <string>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <onnxruntime_cxx_api.h>

#include "runner/detect/YoloDetector.h"
#include "visual/painter.h"

int main(int argc, char const *argv[])
{
    std::cout << "OnnxRuntime verison: " << Ort::GetVersionString() << std::endl;
    std::cout << "Opencv version: " << cv::getVersionString() << std::endl;
    std::string config_path = "config/model_config.yaml";

    std::string img_path = "assets/bus.png";
    try
    {
        std::unique_ptr<Detector> detector = Detector::Load<YoloDetector>(config_path);

        cv::Mat img = cv::imread(img_path);

        auto time_start = std::chrono::high_resolution_clock::now();
        std::vector<Detection> detections = detector->detect(img);
        auto time_end = std::chrono::high_resolution_clock::now();
        
        std::cout << "Time spends: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count() << " ms" << std::endl;
    
        std::cout << "detection num: " << detections.size() << std::endl;

        for (const auto& det : detections) {
            std::cout << detector->class_label(det.class_id) << " " << det.score << std::endl;
            draw_box(img, det, RED);
        }


        // cv::imwrite("output.jpg", img);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    
    
    
    return 0;
}
