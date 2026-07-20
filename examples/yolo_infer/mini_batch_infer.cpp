#include <iostream>
#include <future>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <thread>
#include <stdexcept>

#include "runner/detect/YoloDetector.h"
#include "thread/ThreadPool.h"
#include "logger/logger.h"
#include "ScopedTimer.h"

class BatchInference {
private:
    std::shared_ptr<Detector> detector_;
    ThreadPool pool_;
    int batch_;

    std::queue<cv::Mat> imgs_;
    std::queue<std::shared_ptr<std::promise<std::vector<Detection>>>> promises_;

    bool is_close_;

    std::thread worker_;
    std::mutex mutex_;
    std::condition_variable condition_;

    void work() {
        while (true) {
            std::unique_lock<std::mutex> lock(mutex_);

            bool condition_met = condition_.wait_for(lock, std::chrono::milliseconds(10), [this]{
                return imgs_.size() >= batch_ || is_close_;
            });

            bool has_data = !imgs_.empty();
            bool batch_ready = imgs_.size() >= static_cast<size_t>(batch_);
            bool timed_out_with_data = !condition_met && has_data; // ✅ 纯超时但有积压数据

            // 如果超时，并且有数据，等不起了，直接推理
            if (batch_ready || timed_out_with_data) {
                do {
                    auto tenbuf = detector_->getTensorBuffer();
                    // if (!tenbuf) {
                    //     LOG_ERROR("getTensorBuffer returned null, skipping batch");
                    //     break; // ✅ 安全退出
                    // }
                    auto tenbufPtr = std::make_shared<TensorBuffer>(tenbuf);
                    
                    int size = std::min(batch_, static_cast<int>(imgs_.size()));
                    if (size == 0 || promises_.size() < static_cast<size_t>(size)) {
                        LOG_ERROR("Insufficient imgs or promises, breaking batch loop");
                        break; // ✅ 防止越界
                    }
            
                    // 准备预处理任务
                    std::vector<std::future<void>> pre_futures;
                    std::vector<std::shared_ptr<std::promise<std::vector<Detection>>>> batch_prms;
                    pre_futures.reserve(size);
                    batch_prms.reserve(size);
            
                    auto* row_buf = tenbufPtr.get();
                    for (int i = 0; i < size; ++i) {
                        cv::Mat img = std::move(imgs_.front());
                        imgs_.pop();
                        batch_prms.push_back(std::move(promises_.front()));
                        promises_.pop();
            
                        pre_futures.push_back(pool_.enqueue([this, img = std::move(img), row_buf, i]() {
                            detector_->preprocess(img, *row_buf, i);
                        }));
                    }
            
                    // ✅ 关键修复：推理+后处理 lambda 必须完整 try-catch
                    pool_.enqueue([this, 
                                   pre_futures = std::move(pre_futures),
                                   tenbufPtr,
                                   prms = std::move(batch_prms)]() mutable {
                        try {
                            // 1. 等待预处理（带异常传播）
                            for (auto& fut : pre_futures) {
                                fut.get(); // 异常会被外层 catch 捕获
                            }
            
                            // 2. GPU 推理
                            ModelOutput model_out = detector_->infer(*tenbufPtr);
            
                            // 3. 后处理（不再嵌套 enqueue，避免二次异常丢失）
                            std::vector<std::vector<Detection>> dets = detector_->postprocess(model_out);
            
                            // 4. 安全设置结果
                            for (size_t i = 0; i < prms.size(); ++i) {
                                prms[i]->set_value(std::move(dets[i]));
                            }
                        } catch (...) {
                            // ✅ 兜底：任何异常都传递给所有关联的 promise
                            auto eptr = std::current_exception();
                            for (auto& prm : prms) {
                                try {
                                    prm->set_exception(eptr);
                                } catch (...) {
                                    // promise 可能已被设置过，忽略二次设置异常
                                }
                            }
                            LOG_ERROR("Batch inference failed: exception propagated to {} promises", prms.size());
                        }
                    });
                } while (imgs_.size() >= static_cast<size_t>(batch_));
            }

            // 如果无需推理且已经停止，那么退出
            if (is_close_) {
                break;
            }
        }
    }


public:
    BatchInference(const std::string& config_path) : detector_(Detector::Load<YoloDetector>(config_path)), is_close_(false) {
        worker_ = std::thread(&BatchInference::work, this);
        batch_ = detector_->getBatchSize();
    }

    ~BatchInference() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            is_close_ = true;
        }
        condition_.notify_all();

        if (worker_.joinable()) {
            worker_.join();
        }
    }

    std::future<std::vector<Detection>> submit(const cv::Mat& img);
};

std::future<std::vector<Detection>> BatchInference::submit(const cv::Mat& img) {
    auto p = std::make_shared<std::promise<std::vector<Detection>>>();
    std::future<std::vector<Detection>> fut = p->get_future();

    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (is_close_) {
            LOG_DEBUG_LOC("Throw an exception.");
            throw std::runtime_error("Submit task in BatchInference closed.");
        }

        imgs_.push(img);
        promises_.push(p);
    }

    condition_.notify_one();

    return fut;
}


int main(int argc, char const *argv[])
{

    logger::Init(logger::LOGLEVEL_TRACE);
    
    int num = 10000;

    std::string img_path = "assets/bus.png";
    std::string config_path = "config/model_config.yaml";

    cv::Mat img = cv::imread(img_path);

    BatchInference batchInference(config_path);

    std::vector<std::future<std::vector<Detection>>> futs;
    futs.reserve(num);
    
    ScopedTimer st("runTime");
    for (int i = 0; i < num; ++i) {
        futs.push_back(batchInference.submit(img.clone()));
    }

    int success_count = 0;
    int fail_count = 0;

    for (int i = 0; i < num; ++i) {
        auto res = futs[i].get();
        if (res.size() == 5) ++success_count;
        else {
            ++fail_count;
        }
    }

    std::cout << "Time: " << st.elapsed_ms() << " ms | "
    << "Success: " << success_count << " | Fail: " << fail_count << std::endl;



    // int num = 10000;
    // std::string img_path = "assets/bus.png";
    // std::string config_path = "config/model_config.yaml";

    // cv::Mat img = cv::imread(img_path);
    // BatchInference batchInference(config_path);

    // ScopedTimer st("runTime");
    // int success_count = 0, fail_count = 0;

    // // ✅ 改进1: 不要预分配 10000 个 future，改用滑动窗口控制并发
    // const int MAX_PENDING = 32; // 根据 GPU 显存和 CPU 内存调整
    // std::deque<std::future<std::vector<Detection>>> pending_futs;

    // for (int i = 0; i < num; ++i) {
    //     // ✅ 改进2: 背压控制 - 等待最早的任务完成后再提交新任务
    //     while (pending_futs.size() >= MAX_PENDING) {
    //         auto& oldest = pending_futs.front();
    //         if (oldest.wait_for(std::chrono::seconds(10)) == std::future_status::ready) {
    //             try {
    //                 auto result = oldest.get();
    //                 success_count++;
    //             } catch (const std::exception& e) {
    //                 LOG_ERROR("Task failed: {}", e.what());
    //                 fail_count++;
    //             }
    //             pending_futs.pop_front();
    //         } else {
    //             LOG_ERROR("⚠️ Task TIMEOUT! Skipping...");
    //             fail_count++;
    //             pending_futs.pop_front(); // 超时也要弹出，否则窗口卡死
    //         }
    //     }
        
    //     // ✅ 改进3: 避免重复 clone，使用共享引用（如果 submit 内部只读）
    //     pending_futs.push_back(batchInference.submit(img)); 
    // }

    // // 处理剩余任务
    // while (!pending_futs.empty()) {
    //     auto& fut = pending_futs.front();
    //     if (fut.wait_for(std::chrono::seconds(10)) == std::future_status::ready) {
    //         try {
    //             auto result = fut.get();
    //             success_count++;
    //         } catch (const std::exception& e) {
    //             LOG_ERROR("Tail task failed: {}", e.what());
    //             fail_count++;
    //         }
    //     } else {
    //         LOG_ERROR("⚠️ Tail task TIMEOUT!");
    //         fail_count++;
    //     }
    //     pending_futs.pop_front();
    // }

    // std::cout << "Time: " << st.elapsed_ms() << " ms | "
    //         << "Success: " << success_count << " | Fail: " << fail_count << std::endl;


    return 0;
}
