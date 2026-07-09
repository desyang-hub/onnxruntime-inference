#include "runner/image_restoration/Restorer.h"
#include "runner/image_restoration/NAFNet.h"

// 通过配置文件初始化模型
Restorer::Restorer(const YAML::Node& config) : ModelRunner(config) {

}