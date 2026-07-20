#include "runner/image_restoration/NAFNet.h"
#include "scheduler/AsyncScheduler.h"
#include "logger/logger.h"


#include <filesystem>
namespace fs = std::filesystem;


int main(int argc, char const *argv[])
{
    std::string config_path = "config/Restorer.yaml";
    // auto restorer = Restorer::Load<NAFNet>(config_path);
    std::shared_ptr<Restorer> restorer = std::make_shared<NAFNet>(YAML::LoadFile(config_path));
    AsyncScheduler<Restorer> scheduler(restorer);



    std::string img_path = "assets/seal.png";

    std::vector<std::string> img_paths = {
        "assets/seal0.png",
        "assets/seal1.png"
    };

    int batch = img_paths.size();
    std::vector<cv::Mat> imgs(batch);

    for (int i = 0; i < batch; ++i) {
        imgs[i] = restorer->restoration(cv::imread(img_paths[i]));
    }

    std::string save_dir = "output";
    fs::create_directory(save_dir);

    for (int i = 0; i < imgs.size(); ++i) {
        cv::imwrite( save_dir + "/" + std::to_string(i) + "_seal.png", imgs[i]);
    }

    cv::Mat img = cv::imread(img_paths[0]);
    auto img_out = scheduler.submit(img);
    cv::imwrite(save_dir + "/img_out.jpg", img_out.get()[0]);

    std::cout << "The restormer demo is finish!" << std::endl;
    
    return 0;
}
