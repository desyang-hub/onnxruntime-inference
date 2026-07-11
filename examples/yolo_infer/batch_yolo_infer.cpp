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
#include <chrono>
#include <cstdlib>  // 必须包含这个头文件

#include "runner/detect/YoloDetector.h"
#include "visual/painter.h"

#include "logger/logger.h"
#include "scheduler/SyncScheduler.h"
#include "scheduler/AsyncScheduler.h"

int main(int argc, char const *argv[])
{
    // logger::Init(logger::LOGLEVEL_TRACE); // 开启才会打印TRACE
    LOG_TRACE("step");
    std::cout << "OnnxRuntime verison: " << Ort::GetVersionString() << std::endl;
    std::cout << "Opencv version: " << cv::getVersionString() << std::endl;
    std::string config_path = "config/model_config.yaml";

    size_t batch = 4;
    std::vector<cv::Mat> imgs;
    imgs.reserve(batch);
    size_t cnt = 10;

    std::string img_path = "assets/bus.png";
    try
    {
        // std::unique_ptr<Detector> detector = Detector::Load<YoloDetector>(config_path);

        
        std::shared_ptr<Detector> detector = std::make_shared<YoloDetector>(YAML::LoadFile(config_path));

        cv::Mat img = cv::imread(img_path);

        for (int i = 0; i < batch; ++i) {
            imgs.push_back(img.clone());
        }

        
        SyncScheduler<Detector> scheduler(detector);
        // AsyncScheduler<Detector> scheduler(detector);

        for (int i = 0; i < batch; ++i) {
            std::future<Detector::OutputType> output = scheduler.submit(img);

            std::vector<Detection> dets = output.get();

            std::cout << "detection nums: " << dets.size() << std::endl;

            for (const auto& det : dets) {
                std::cout << detector->class_label(det.class_id) << " " << det.score << std::endl;
            }
        }

        


        // auto time_start = std::chrono::high_resolution_clock::now();
        // std::vector<std::vector<Detection>> results = detector->detect(imgs);
        // auto time_end = std::chrono::high_resolution_clock::now();

        // std::cout << "Time spends: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count() << " ms" << std::endl;

        // std::cout << std::endl;

        
        // std::cout << results.size() << " results num" << std::endl;

        // for (auto& result : results) {
        //     std::cout << result.size() << " det size" << std::endl;
        //     for (const auto& det : result) {
        //         std::cout << detector->class_label(det.class_id) << " " << det.score << std::endl;
        //         draw_box(img, det, RED);
        //     }
        // }

        std::cout << "inference ending" << std::endl;

        // cv::imwrite("output.jpg", img);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    
    std::cout << "program finish" << std::endl;
    
    return 0;
}
