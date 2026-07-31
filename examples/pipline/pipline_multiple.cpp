#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "pipline/PipLine.h"
#include "ScopedTimer.h"
#include "pipline/SafeQueuePipLine.h"

int main(int argc, char const *argv[])
{
    std::string config_path = "config/model_config.yaml";
    std::string img_path = "assets/bus.png";

    using PipLineType = SafeQueuePipLine;

    cv::Mat img = cv::imread(img_path);
    auto config = YAML::LoadFile(config_path);
    PipLineType detector(config);

    int instance = 1;
    int num = 1000;

    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "pipline");

    std::vector<std::unique_ptr<PipLineType>> piplines;
    piplines.reserve(instance);

    for (int i = 0; i < instance; ++i) {
        piplines.push_back(std::make_unique<PipLineType>(config, env));
    }

    std::vector<std::future<std::vector<Detection>>> futs;
    futs.reserve(num);

    auto res = detector.submit(img).get();
    std::cout << "object num: " << res.size() << std::endl;
    // res = detector.submit(img).get();
    // std::cout << "object num: " << res.size() << std::endl;
    // res = detector.submit(img).get();
    // std::cout << "object num: " << res.size() << std::endl;


    ScopedTimer st("pipline RunTime");
    for (int i = 0; i < num; ++i) {
        futs.push_back(piplines[i % piplines.size()]->submit(img));
    }
    for (int i = 0; i < num; ++i) {
        futs[i].get();
    }

    std::cout << "Total Sepnds: " << st.elapsed_ms() << std::endl;
    
    return 0;
}
