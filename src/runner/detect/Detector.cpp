#include "runner/detect/Detector.h"

Detector::Detector(const YAML::Node& config) : ModelRunner(config) 
{
    labels_ = config["classes"].as<std::vector<std::string>>();
}

const std::string& Detector::class_label(size_t id) const {
    return labels_[id];
}