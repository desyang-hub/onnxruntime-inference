#include "runner/image_restoration/NAFNet.h"

int main(int argc, char const *argv[])
{
    
    std::string config_path = "config/Restorer.yaml";
    auto restorer = Restorer::Load<NAFNet>(config_path);

    std::string img_path = "assets/seal.png";

    std::vector<std::string> img_paths = {
        "assets/seal0.png",
        "assets/seal1.png"
    };

    int batch = img_paths.size();
    std::vector<cv::Mat> imgs;
    imgs.reserve(batch);

    for (int i = 0; i < batch; ++i) {
        imgs.push_back(cv::imread(img_paths[i]));
    }
    
    imgs = restorer->restoration(imgs);

    for (int i = 0; i < imgs.size(); ++i) {
        cv::imwrite("assets/" + std::to_string(i) + "_seal.png", imgs[i]);
    }

    return 0;
}
