#include "scheduler/BatchScheduler.h"
#include "runner/detect/YoloDetector.h"
#include "runner/image_restoration/NAFNet.h"
#include "ScopedTimer.h"

int main(int argc, char const *argv[])
{
    // std::string config_path = "config/model_config.yaml";
    // std::shared_ptr<Detector> detector = Detector::Load<YoloDetector>(config_path);

    // auto scheduler = std::make_shared<BatchScheduler<Detector>>(detector);

    std::string config_path = "config/Restorer.yaml";
    std::shared_ptr<Restorer> runner = Restorer::Load<NAFNet>(config_path);

    auto scheduler = std::make_shared<BatchScheduler<Restorer>>(runner);


    int num = 100;
    std::string img_path = "assets/0_seal.png";
    cv::Mat img = cv::imread(img_path);

    std::vector<std::future<Restorer::OutputType>> futs;
    futs.reserve(num);
    
    ScopedTimer st("batchScheduler");
    for (int i = 0; i < num; ++i) {
        futs.push_back(scheduler->submit(img.clone()));
    }

    int acc{};
    int failed{};

    for (int i = 0; i < num; ++i) {
        auto res = futs[i].get();
    }

    // std::cout << "success: " << acc << ", failed: " << failed << std::endl;
    std::cout << "Prgram always quit successful!" << std::endl;
    

    return 0;
}
