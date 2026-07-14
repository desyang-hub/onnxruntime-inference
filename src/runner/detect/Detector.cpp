#include "runner/detect/Detector.h"

#include "ScopedTimer.h"
#include "logger/logger.h"

Detector::Detector(const YAML::Node& config) : ModelRunner(config) 
{
    labels_ = config["classes"].as<std::vector<std::string>>();
}

std::vector<Detection> Detector::detect(const cv::Mat& img) {
    // ScopedTimer timer("detect");
    TensorBuffer pre_out = preprocess(img);
    // LOG_INFO("preprocess: {} ms", timer.elapsed_ms());
    ModelOutput infer_out = infer(pre_out);
    // LOG_INFO("infer: {} ms", timer.elapsed_ms());
    auto res = postprocess(infer_out)[0];
    // LOG_INFO("postprocess: {} ms", timer.elapsed_ms());
    return res;
}

const std::string& Detector::class_label(size_t id) const {
    return labels_[id];
}