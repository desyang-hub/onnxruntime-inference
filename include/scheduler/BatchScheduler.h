#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <stdexcept>

#include "InferenceScheduler.h"
#include "thread/ThreadPool.h"
#include "stage/Stage.h"
#include "logger/logger.h"
#include "runner/ModelRunner.h"

static constexpr int kTimeoutMiliSeconds = 5; // ms

// template<class Runner = ModelRunner>
template<class Runner>
class BatchScheduler : InferenceScheduler<Runner>
{
    using InputType = typename runner_type_trait<Runner>::InputType;
    using OutputType = typename runner_type_trait<Runner>::OutputType;
private:
    std::shared_ptr<Runner> backend_;
    ThreadPool pool_;
    int batch_;

    std::queue<InputType> inputs_;
    std::queue<std::shared_ptr<std::promise<OutputType>>> promises_;

    bool is_close_;

    std::thread worker_;
    std::mutex mutex_;
    std::condition_variable condition_;

    std::unique_ptr<PreStage<Runner>>     pre_stage_;     // 预处理
    std::unique_ptr<Stage<TensorBuffer, ModelOutput>>  infer_stage_;    // 推理器
    std::unique_ptr<Stage<ModelOutput, std::vector<OutputType>>>    post_stage_;     // 后处理

    int buffer_size_;
    ThreadPool pre_executor_;
    ThreadPool infer_executor_;
    ThreadPool post_executor_;

    void run() {
        while (true) {
            std::unique_lock<std::mutex> lock(mutex_);

            // 检测是否超时或者batch满了
            bool condition_met = condition_.wait_for(lock, std::chrono::milliseconds(kTimeoutMiliSeconds), [this]{
                return inputs_.size() >= batch_ || is_close_;
            });

            bool has_data = !inputs_.empty();
            bool batch_ready = inputs_.size() >= static_cast<size_t>(batch_);
            bool timed_out_with_data = !condition_met && has_data; // 超时且有数据

            // batch_ready 或 超时有数据都进入推理
            if (batch_ready || timed_out_with_data) {
                do {
                    TensorBuffer tenbuf = backend_->getTensorBuffer();

                    if (!tenbuf.valid())  {
                        LOG_ERROR("getTensorBuffer returned null, skipping batch !");
                        break;
                    }
                    
                    auto tenbufPtr = std::make_shared<TensorBuffer>(tenbuf);
                    
                    int size = std::min(batch_, static_cast<int>(inputs_.size()));
                    if (size == 0 || promises_.size() < static_cast<size_t>(size)) {
                        LOG_ERROR("Insufficient imgs or promises, breaking batch loop");
                        break; // ✅ 防止越界
                    }
            
                    // 准备预处理任务
                    std::vector<std::future<void>> pre_futures;
                    std::vector<std::shared_ptr<std::promise<OutputType>>> batch_prms;
                    pre_futures.reserve(size);
                    batch_prms.reserve(size);
            
                    auto row_buf = tenbufPtr.get();
                    for (int i = 0; i < size; ++i) {
                        InputType img = std::move(inputs_.front());
                        inputs_.pop();
                        batch_prms.push_back(std::move(promises_.front()));
                        promises_.pop();
            
                        pre_futures.push_back(pre_executor_.enqueue([this, img = std::move(img), row_buf, i]() {
                            pre_stage_->execute(img, *row_buf, i);
                        }));
                    }
            
                    // ✅ 关键修复：推理+后处理 lambda 必须完整 try-catch
                    infer_executor_.enqueue([this, 
                                   pre_futures = std::move(pre_futures),
                                   tenbufPtr,
                                   prms = std::move(batch_prms)]() mutable 
                    {
                        try 
                        {
                            // 1. 等待预处理（带异常传播）
                            for (auto& fut : pre_futures) {
                                fut.get(); // 异常会被外层 catch 捕获
                            }
            
                            // 2. GPU 推理
                            ModelOutput model_out = infer_stage_->execute(*tenbufPtr);

                            post_executor_.enqueue([this, model_out = std::move(model_out), 
                                prms = std::move(prms)]
                            {
                                try {
                                    // 3. 后处理（不再嵌套 enqueue，避免二次异常丢失）
                                    std::vector<OutputType> outputs = post_stage_->execute(model_out);

                                    // 4. 安全设置结果
                                    for (size_t i = 0; i < prms.size(); ++i) {
                                        prms[i]->set_value(std::move(outputs[i]));
                                    }
                                }
                                catch (...) {
                                    auto eptr = std::current_exception();
                                    for (auto& prm : prms) {
                                        try {
                                            prm->set_exception(eptr);
                                        } catch (...) {
                                            // promise 可能已被设置过，忽略二次设置异常
                                        }
                                    }
                                }
                            });
                            
                        } 
                        catch (...) 
                        {
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
                } while (inputs_.size() >= static_cast<size_t>(batch_));
            }

            // 如果无需推理且已经停止，那么退出
            if (is_close_) {
                break;
            }
        }
    }

public:
    BatchScheduler(std::shared_ptr<Runner> backend) :
        backend_(backend),
        is_close_(false),
        buffer_size_(backend->getBufferSize()),
        pre_executor_(buffer_size_),
        infer_executor_(buffer_size_),
        post_executor_(buffer_size_),
        pre_stage_(std::make_unique<PreStage<Runner>>(backend)),
        infer_stage_(std::make_unique<InferStage<Runner>>(backend)),
        post_stage_(std::make_unique<PostStage<Runner>>(backend))
    {
        batch_ = backend->getBatchSize();
        worker_ = std::thread(&BatchScheduler::run, this);
    }

    ~BatchScheduler() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            is_close_ = true;
        }
        condition_.notify_all();

        if (worker_.joinable()) {
            worker_.join();
        }
    }

    std::future<OutputType> submit(const InputType& input) override {
        auto p = std::make_shared<std::promise<OutputType>>();
        std::future<OutputType> fut = p->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (is_close_) {
                LOG_DEBUG("submit Task in Scheduler quit!");
                throw std::runtime_error("submit Task in Scheduler quit!");
            }

            inputs_.push(input);
            promises_.push(p);
        }

        condition_.notify_one();

        return fut;
    }
};