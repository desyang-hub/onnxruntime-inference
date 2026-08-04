#include "runner/OnnxRunner.h"
#include "thread/ThreadPool.h"
#include "ScopedTimer.h"

// Spends 20152.7 ms
int main(int argc, char const *argv[])
{
    int num = 100;
    std::string img_path = "assets/bus.png";
    cv::Mat img = cv::imread(img_path);
    std::string config_path = "config/Yolo.yaml";


    OnnxRunner detector(YAML::LoadFile(config_path));

    auto res = detector.detect(img);

    std::cout << res[0].size() << std::endl;
    for (auto& item : res[0]) {
        std::cout << item.class_id << " " << item.score << std::endl;
    }

    ScopedTimer st("MultiSession");
    for (int i = 0; i < num; ++i) {
        detector.detect(img);
    }
    std::cout << "Spends " << st.elapsed_ms() << " ms" << std::endl;

    return 0;
}
