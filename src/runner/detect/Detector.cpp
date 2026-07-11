#include "runner/detect/Detector.h"

Detector::Detector(const YAML::Node& config) : ModelRunner(config) 
{
    labels_ = config["classes"].as<std::vector<std::string>>();
}

std::vector<Detection> Detector::detect(const cv::Mat& img) {
    TensorBuffer pre_out = preprocess(img);
    ModelOutput infer_out = infer(pre_out);
    return postprocess(infer_out)[0];
}

const std::string& Detector::class_label(size_t id) const {
    return labels_[id];
}