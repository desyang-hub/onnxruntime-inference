#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <stdint.h>
#include <unordered_map>
#include <stdexcept>
#include <memory>

#include <cuda_runtime.h>
#include "nonecopyable.h"

/// @brief 推理张量缓存池，管理GPU内存
class InferTensorBufferPool : public nonecopyable
{
private:
    void* data_;
    std::queue<void*> buffer_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    bool is_close;
    size_t capacity_;


    // 初始化空缓冲
    void init(size_t ele_size, size_t ele_bytes_size) {
        for (int i = 0; i < ele_size; ++i) {
            buffer_.push(static_cast<char*>(data_) + i * ele_bytes_size);
        }
    }

public:
    /// @brief 用于初始化缓存池
    /// @param ele_size 元素个数
    /// @param ele_bytes_size 单个元素所占字节数
    InferTensorBufferPool(size_t ele_size, size_t ele_bytes_size) : data_(nullptr), is_close(false), capacity_(ele_size) {
        cudaMalloc(&data_, ele_size * ele_bytes_size);
        init(ele_size, ele_bytes_size);
    }

    ~InferTensorBufferPool() {
        this->close();
    }

    size_t capacity() const {
        return capacity_;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return buffer_.size();
    }

    // 返回可用的内存指针, 阻塞获取
    template<class T = float>
    T* Acquire() {
        std::unique_lock<std::mutex> lock(mutex_);

        condition_.wait(lock, [this]{
            return !buffer_.empty() || is_close;
        });

        if (!buffer_.empty()) {
            void* data = buffer_.front();
            buffer_.pop();
            return reinterpret_cast<T*>(data);
        }

        return nullptr;
    }

    void Release(void* data) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (is_close) {
            throw std::runtime_error("Release discover in Pool closed");
        }

        
        if (data == nullptr) {
            throw std::runtime_error("释放的指针数据为空");
        }

        buffer_.push(data);
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (is_close) return;
            is_close = true;
        }
        condition_.notify_all();

        // 归还GPU数据
        cudaFree(data_);
    }
};

using InferTensorBufferPoolPtr = std::unique_ptr<InferTensorBufferPool>;