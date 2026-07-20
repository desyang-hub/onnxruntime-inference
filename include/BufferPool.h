#pragma once

#include <queue>
#include <mutex>
#include <memory>
#include <condition_variable>

/**
 * @brief 实现一个缓冲池
 * 
 * @note 缓冲池的作用就是一次初始化一大片空间
 * 
 */
class BufferPool
{
private:
    /* data */
    std::unique_ptr<char[]> data_;
    std::queue<void*> que_;

    std::mutex mutex_;
    std::condition_variable condition_;
    bool is_close_ = false;
    size_t total_bytes_;

    struct buffer_pool_deleter {
        BufferPool& pool_;
    
        void operator()(void* ptr) { 
            {
                std::lock_guard<std::mutex> lock(pool_.mutex_);
                if (ptr == nullptr) return;
                pool_.que_.push(ptr);
            }
            pool_.condition_.notify_one();
        }
    };

public:
    BufferPool(int buffer_size, int element_bytes_size);
    ~BufferPool();

    BufferPool(const BufferPool&) = delete;
    BufferPool& operator=(const BufferPool&) = delete;

    template<class T = float>
    std::shared_ptr<T> Acquire();
};

template<class T>
std::shared_ptr<T> BufferPool::Acquire() { 

    std::unique_lock<std::mutex> lock(mutex_);

    condition_.wait(lock, [this]{
        return !que_.empty() || is_close_;
    });

    if (!que_.empty()) {
        T* ptr = reinterpret_cast<T*>(que_.front());
        que_.pop();
        return std::shared_ptr<T>(ptr, buffer_pool_deleter{*this});
    }

    return nullptr;
}

using BufferPoolPtr = std::unique_ptr<BufferPool>;