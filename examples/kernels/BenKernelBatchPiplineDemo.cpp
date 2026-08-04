#include "kernels/BenchKernelBatchPipline.h"



int main(int argc, char const *argv[])
{
    int num = 10000;
    std::string config_path = "config/model_config.yaml";
    std::string img_path = "assets/bus.png";

    cv::Mat img = cv::imread(img_path);
    YAML::Node config = YAML::LoadFile(config_path);

    // BenchKernel BenchBase
    std::unique_ptr<BenchKernelBatchPipline> bench = std::make_unique<BenchKernelBatchPipline>(config);

    int batch_size = bench->getBatchSize();
    int num_batch =  (num + batch_size - 1) / batch_size;

    auto res = bench->submit(img).get();

    LOG_INFO("res size: {}", res.size());

    for (auto& item : res) {
        std::cout << item.class_id << " " << item.score << std::endl;
    }

    // std::vector<std::vector<Detection>> res;
    // res.reserve(num_batch);
    // std::vector<std::future<Detector::OutputType>> futs;
    // futs.reserve(num_batch);
    // ScopedTimer st("");
    
    // for (int i = 0; i < num; ++i) {
    //     futs.push_back(bench->submit(img));
    // }
    // for (int i = 0; i < num; ++i) {
    //     res.push_back(futs[i].get());
    // }

    // std::cout << "BenchBase spends: " << st.elapsed_ms() << std::endl;
    // std::cout << res[0].size() << std::endl;
    // for (int i = 0; i < res[0].size(); ++i) {
    //     std::cout << res[0][i].class_id << " " << res[0][i].score << std::endl;
    // }
    
    
    return 0;
}
