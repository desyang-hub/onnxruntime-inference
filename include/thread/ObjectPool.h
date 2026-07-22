#pragma once

#include <queue>
#include <mutex>
#include <functional>
#include <stdexcept>
#include <condition_variable>

template<class T>
class ObjectPool
{
private:
    std::queue<T*> que_;
    std::mutex mutex_;
    std::condition_variable condition_;
    bool is_close_;

    std::function<void(T*)> destroy_callback_;

    struct object_pool_retriever {
        ObjectPool& self;

        void operator()(T* p) {
            if (!p) return;

            // 释放逻辑
            self.Release(p);
        }
    };

public:
    ObjectPool() : is_close_(false), destroy_callback_{} {

    }

    // 析构过程触发回调
    ~ObjectPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            is_close_ = true;
        }
        condition_.notify_all();

        // 可以设置回调来清理资源，不过一般而言，资源一般是谁new谁处理
        if (destroy_callback_) {
            while (!que_.empty()) {
                destroy_callback_(que_.front());
                que_.pop();
            }
        }
    }

    // 设置回调
    void SetDestroyCallback(const std::function<void(T*)>& cb) {
        destroy_callback_ = cb;
    }

    void SetDestroyCallback(std::function<void(T*)>&& cb) {
        destroy_callback_ = std::move(cb);
    }

    void Release(T* p) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            
            if (is_close_) throw std::runtime_error("Release resource in stoped!");

            que_.push(p);
        }

        condition_.notify_one();
    }

    std::unique_ptr<T, object_pool_retriever> AcquireUnique() {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this]{
            return !que_.empty() || is_close_;
        });

        if (!que_.empty()) {
            auto p = que_.front();
            que_.pop();
            return std::unique_ptr<T, object_pool_retriever>(p, object_pool_retriever{*this});
        }

        return std::unique_ptr<T, object_pool_retriever>(nullptr, object_pool_retriever{*this});
    }

    std::shared_ptr<T> Acquire() {
        std::unique_lock<std::mutex> lock(mutex_);

        condition_.wait(lock, [this]{
            return !que_.empty() || is_close_;
        });

        if (!que_.empty()) {
            auto p = que_.front();
            que_.pop();
            return std::shared_ptr<T>(p, object_pool_retriever{*this});
        }

        return nullptr;
    }
};