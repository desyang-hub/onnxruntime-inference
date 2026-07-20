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
#include "logger/logger.h"

/**
 * @brief 推理张量缓存池，管理GPU内存
 * 
 * @note 目前用完需要手动释放，后续可以修改为用完自动回收
 */
class InferTensorBufferPool : public nonecopyable
{
private:
    void*       data_;
    size_t      total_bytes_;
    size_t      ele_bytes_size_;
    std::queue<void*> buffer_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    bool        is_closed_;
    size_t      capacity_;

    // 用于 Release 时的合法性校验
    inline bool owns(void* ptr) const noexcept {
        auto p = reinterpret_cast<uintptr_t>(ptr);
        auto base = reinterpret_cast<uintptr_t>(data_);
        if (p < base || p >= base + total_bytes_) return false;
        return (p - base) % ele_bytes_size_ == 0;  // 必须是对齐的合法偏移
    }

public:
    InferTensorBufferPool(size_t ele_size, size_t ele_bytes_size)
        : data_(nullptr), total_bytes_(ele_size * ele_bytes_size),
          ele_bytes_size_(ele_bytes_size), is_closed_(false), capacity_(ele_size)
    {
        if (ele_size == 0 || ele_bytes_size == 0)
            throw std::invalid_argument("Pool size and element size must be > 0");

        cudaError_t err = cudaMalloc(&data_, total_bytes_);
        if (err != cudaSuccess || data_ == nullptr) {
            throw std::runtime_error(
                std::string("cudaMalloc failed in InferTensorBufferPool: ") 
                + cudaGetErrorString(err));
        }

        for (size_t i = 0; i < ele_size; ++i) {
            buffer_.push(static_cast<char*>(data_) + i * ele_bytes_size_);
        }
    }

    ~InferTensorBufferPool() { close(); }

    size_t capacity() const noexcept { return capacity_; }

    size_t available() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return buffer_.size();
    }

    template<class T = float>
    T* Acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this]{ return !buffer_.empty() || is_closed_; });

        if (is_closed_) return nullptr;  // ✅ 关闭后统一返回 nullptr

        void* ptr = buffer_.front();
        buffer_.pop();
        return reinterpret_cast<T*>(ptr);
    }

    void Release(void* ptr) {
        if (ptr == nullptr) return;  // 允许 release nullptr，幂等

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (is_closed_) {
                // ✅ 关闭后不再 push，也不抛异常（优雅降级）
                LOG_WARN("Release called after pool closed, ignoring.");
                return;
            }
            if (!owns(ptr)) {
                LOG_ERROR("Release: pointer not owned by this pool! Ignoring. ptr: {}", fmt::ptr(ptr));
                throw std::runtime_error("Release not owned py this poll!");
            }
            buffer_.push(ptr);
        }
        condition_.notify_one();
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (is_closed_) return;
            is_closed_ = true;
            // ✅ 不在锁内 cudaFree，先唤醒所有等待者
        }
        condition_.notify_all();

        // ✅ 安全释放：此时没有新 Acquire 能拿到有效指针
        // 注意：已 Acquire 但未 Release 的指针由调用方负责
        // 如需等待全部归还，需引入引用计数或 active_count
        if (data_) {
            cudaFree(data_);
            data_ = nullptr;
        }
    }
};

using InferTensorBufferPoolPtr = std::unique_ptr<InferTensorBufferPool>;