#include "runner/classify/Classifier.h"

Classifier::Classifier(const YAML::Node& config) : ModelRunner(config) {
    labels_ = config["classes"].as<std::vector<std::string>>();
}

int Classifier::classify(const cv::Mat& img) {
    auto pre_out = preprocess(img);
    auto model_output = infer(pre_out);
    return postprocess(model_output)[0];
}

const std::string& Classifier::class_label(size_t id) const {
    return labels_[id];
}