#include "kernels/BenchBase.h"
#include "kernels/BenchKernel.h"

#include "ScopedTimer.h"

int main(int argc, char const *argv[])
{
    
    std::string config_path = "config/model_config.yaml";
    std::string img_path = "assets/bus.png";

    cv::Mat img = cv::imread(img_path);
    YAML::Node config = YAML::LoadFile(config_path);
    std::unique_ptr<Bench> bench = std::make_unique<BenchBase>(config);

    // ===================== BenchBase =====================
    ScopedTimer st("detect");
    auto res =  bench->detect(img);
    std::cout << "BenchBase spends: " << st.elapsed_ms() << std::endl;
    std::cout << res.size() << std::endl;

    
    // ===================== BenchKernel =====================
    bench = std::make_unique<BenchKernel>(config);
    ScopedTimer st1("detect");
    res =  bench->detect(img);
    std::cout << "BenchKernel spends: " << st1.elapsed_ms() << std::endl;
    std::cout << res.size() << std::endl;

    for (int i = 0; i < res.size(); ++i) {
        std::cout << res[i].class_id << " " << res[i].score << std::endl;
    }

    return 0;
}
