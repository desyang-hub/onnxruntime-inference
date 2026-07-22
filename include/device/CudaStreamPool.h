#pragma once

#include <queue>
#include <memory>
#include <mutex>
#include <condition_variable>

#include <cuda_runtime.h>

class CudaStreamPool
{
private:
    std::queue<cudaStream_t> que_;
    std::mutex mutex_;
    std::condition_variable condition_;
    bool is_close_;

    struct cuda_stream_deleter {
        CudaStreamPool& self;

        void operator()(cudaStream_t p) const {
            if (!p) return;
            self.Release(p);
        }
    };

    void Release(cudaStream_t p) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            que_.push(p);
        }
        condition_.notify_one();
    }

public:
    explicit CudaStreamPool(int size) : is_close_(false) {
        for (int i = 0; i < size; ++i) {
            cudaStream_t stream{};
            cudaStreamCreate(&stream);
            que_.push(stream);
        }
    }

    ~CudaStreamPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            is_close_ = true;
        }
        condition_.notify_all();

        // 将资源释放掉
        while (!que_.empty()) {
            cudaStreamDestroy(que_.front());
            que_.pop();
        }
    }

    std::shared_ptr<cudaStream_t> Acquire() {
        std::unique_lock<std::mutex> lock(mutex_);

        condition_.wait(lock, [this]{
            return !que_.empty() || is_close_;
        });
        
        if (!que_.empty()) {
            cudaStream_t p = que_.front();
            que_.pop();
            return std::make_shared<cudaStream_t>(p, cuda_stream_deleter{*this});
        }

        return nullptr;
    }
};
