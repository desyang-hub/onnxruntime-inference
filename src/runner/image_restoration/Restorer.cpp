#include "runner/image_restoration/Restorer.h"
#include "runner/image_restoration/NAFNet.h"
#include "logger/logger.h"

// 通过配置文件初始化模型
Restorer::Restorer(const YAML::Node& config) : ModelRunner(config) {

}

cv::Mat Restorer::restoration(const cv::Mat& img) {
    auto pre_out = preprocess(img);
    auto model_out = infer(pre_out);
    return postprocess(model_out)[0];
}