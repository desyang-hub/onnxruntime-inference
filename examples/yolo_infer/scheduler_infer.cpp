#include <string>
#include <opencv2/opencv.hpp>

#include "scheduler/SyncScheduler.h"
#include "scheduler/AsyncScheduler.h"
#include "scheduler/BatchScheduler.h"

#include "runner/detect/YoloDetector.h"
#include "ScopedTimer.h"


/**
 * num: 10000 image
 * SyncScheduler:   23931.8 ms
 * AsyncScheduler:  12393.2 ms
 * BatchScheduler:  9441.68 ms
 * 
 * 
 */

int main(int argc, char const *argv[])
{
    // std::string config_path = "config/sync_yolo_config.yaml";
    std::string config_path = "config/async_yolo_config.yaml";
    // std::string config_path = "config/batch_yolo_config.yaml";

    std::string img_path = "assets/bus.png";
    cv::Mat img = cv::imread(img_path);

    int num = 10000;

    auto detector = Detector::Load<YoloDetector>(config_path);

    // auto scheduler = std::make_shared<SyncScheduler<Detector>>(detector);
    auto scheduler = std::make_shared<AsyncScheduler<Detector>>(detector);
    // auto scheduler = std::make_shared<BatchScheduler<Detector>>(detector);

    std::vector<std::future<Detector::OutputType>> futs;
    futs.reserve(num);

    ScopedTimer st("schedulerTimer");
    for (int i = 0; i < num; ++i) {
        futs.push_back(scheduler->submit(img));
    }

    for (int i = 0; i < num; ++i) {
        futs[i].get();
    }

    std::cout << "Total num: " << num << ", Spends " << st.elapsed_ms() << " ms" << std::endl;

    return 0;
}
