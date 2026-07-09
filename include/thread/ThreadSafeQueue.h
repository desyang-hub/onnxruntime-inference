#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>

template<class T>
class ThreadSafeQueue
{
    static constexpr size_t kUnbounded = 0;
private:
    std::queue<T> queue_;
    size_t capacity_; // 0 表示无限
    std::mutex mutex_;
    std::condition_variable not_full_;
    std::condition_variable not_empty_;
    bool is_close_;

public:
    explicit ThreadSafeQueue(size_t capacity);
    ~ThreadSafeQueue();

    ThreadSafeQueue(const ThreadSafeQueue&) = delete;
    ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;
    ThreadSafeQueue(ThreadSafeQueue&&) = delete;
    ThreadSafeQueue& operator=(ThreadSafeQueue&&) = delete;

    bool Push(const T& value);
    bool Push(T&& value);

    bool Pop(T& value);
    bool TryPop(T& value);

    void Close();
};

template<class T>
ThreadSafeQueue<T>::ThreadSafeQueue(size_t capacity) : capacity_(capacity), is_close_(false) {

}

template<class T>
ThreadSafeQueue<T>::~ThreadSafeQueue() {
    Close();
}

template<class T>
bool ThreadSafeQueue<T>::Push(const T& value) {
    {
        std::unique_lock<std::mutex> lock(mutex_);

        if (capacity_ != kUnbounded && queue_.size() == capacity_) {
            not_full_.wait(lock, [this]{
                return queue_.size() < capacity_ || is_close_;
            });
        }

        if (is_close_) return false;

        queue_.push(value);
    }

    not_empty_.notify_one();
    
    return true;
}

template<class T>
bool ThreadSafeQueue<T>::Push(T&& value) {
    {
        std::unique_lock<std::mutex> lock(mutex_);

        if (capacity_ != kUnbounded && queue_.size() == capacity_) {
            not_full_.wait(lock, [this]{
                return queue_.size() < capacity_ || is_close_;
            });
        }

        if (is_close_) return false;

        queue_.push(std::move(value));        
    }
    not_empty_.notify_one();
    return true;
}

template<class T>
bool ThreadSafeQueue<T>::Pop(T& value) {
    {
        std::unique_lock<std::mutex> lock(mutex_);

        // 进入等待
        if (queue_.empty()) {
            not_empty_.wait(lock, [this]{
                return !queue_.empty() || is_close_;
            });

            if (queue_.empty() && is_close_) {
                return false;
            }
        }

        value = std::move(queue_.front());
        queue_.pop();
    }

    not_full_.notify_one();
    return true;
}

template<class T>
bool ThreadSafeQueue<T>::TryPop(T& value) {
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (queue_.empty()) {
            return false;
        }

        value = std::move(queue_.front());
        queue_.pop();
    }

    not_full_.notify_one();
    return true;
}

template<class T>
void ThreadSafeQueue<T>::Close() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        is_close_ = true;
    }
    not_empty_.notify_all();
    not_full_.notify_all();
}