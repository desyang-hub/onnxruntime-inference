#include "kernels/BenchBase.h"
#include "kernels/BenchKernel.h"
#include "kernels/BenchKernelBatch.h"
#include "pipline/TaskContext.h"

#include "ScopedTimer.h"
#include "thread/ThreadPool.h"


void compare_spend_times() {
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
}

void sync_test_demo() {
    int num = 500;
    std::string config_path = "config/model_config.yaml";
    std::string img_path = "assets/bus.png";

    cv::Mat img = cv::imread(img_path);
    YAML::Node config = YAML::LoadFile(config_path);
    // BenchKernel BenchBase
    std::unique_ptr<Bench> bench = std::make_unique<BenchKernel>(config);

    ScopedTimer st("");
    for (int i = 0; i < num; ++i) {
        bench->detect(img);
    }

    std::cout << "BenchBase spends: " << st.elapsed_ms() << std::endl;
}

int main(int argc, char const *argv[])
{
    int num = 10000;
    std::string config_path = "config/model_config.yaml";
    std::string img_path = "assets/bus.png";

    cv::Mat img = cv::imread(img_path);
    YAML::Node config = YAML::LoadFile(config_path);

    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "test");
    TaskContext context(config, env);
    int batch_size = context.batch_size;
    int num_batch =  (num + batch_size - 1) / batch_size;

    // BenchKernel BenchBase
    std::unique_ptr<BenchKernelBatch> bench = std::make_unique<BenchKernelBatch>(std::move(context));

    std::vector<cv::Mat> imgs;
    imgs.reserve(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        imgs.push_back(img);
    }

    ScopedTimer st("");
    std::vector<std::vector<Detection>> res;
    for (int i = 0; i < num_batch; ++i) {
        res = bench->batch_detect(imgs);
    }
    std::cout << "BenchBase spends: " << st.elapsed_ms() << std::endl;
    std::cout << res[0].size() << std::endl;
    for (int i = 0; i < res[0].size(); ++i) {
        std::cout << res[0][i].class_id << " " << res[0][i].score << std::endl;
    }

    return 0;
}
