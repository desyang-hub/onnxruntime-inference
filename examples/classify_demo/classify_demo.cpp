#include "runner/classify/ResNet.h"
#include "scheduler/AsyncScheduler.h"
#include "logger/logger.h"

#include <yaml-cpp/yaml.h>


#include <filesystem>
namespace fs = std::filesystem;


int main(int argc, char const *argv[])
{
    std::string config_path = "config/Classify.yaml";

    std::cout << YAML::LoadFile(config_path) << std::endl;

    ResNet net(YAML::LoadFile(config_path));

    std::string img_path = "assets/cat.png";
    // std::string img_path = "assets/imgnet.jpg";

    int d = net.classify(cv::imread(img_path));

    std::cout << net.class_label(d) << std::endl;
    
    
    return 0;
}
