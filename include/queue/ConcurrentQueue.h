#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <stdexcept>

template<class T>
class ConcurrentQueue
{
private:

    std::queue<T> que_;
    std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    bool is_close_{false};
    size_t capacity_{0}; // 0表示无上限

    bool isFull() const {
        return capacity_ && que_.size() >= capacity_;
    }
public:
    ConcurrentQueue(size_t capacity = 0) : capacity_(capacity) {

    }

    ~ConcurrentQueue() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            is_close_ = true;
        }
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    ConcurrentQueue(const ConcurrentQueue&) = delete;
    ConcurrentQueue& operator=(const ConcurrentQueue&) = delete;

    void push(const T& item) {
        std::unique_lock<std::mutex> lock(mutex_);

        if (is_close_) {
            throw std::runtime_error("Push operation be found in que stoped.");
        }

        not_full_.wait(lock, [this](){
            return !isFull();
        });

        que_.push(item);
        not_empty_.notify_one();
    }

    void push(T&& item) {
        std::unique_lock<std::mutex> lock(mutex_);

        if (is_close_) {
            throw std::runtime_error("Push operation be found in que stoped.");
        }

        not_full_.wait(lock, [this](){
            bool full = isFull();

            return !isFull();
        });

        que_.push(std::move(item));
        not_empty_.notify_one();
    }

    void pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this]{
            return !que_.empty() || is_close_;
        });

        if (!que_.empty()) {
            item = std::move(que_.front());
            que_.pop();
            not_full_.notify_one();
        }
        else if (is_close_) {
            return;
        }
    }

    T pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this]{
            return !que_.empty() || is_close_;
        });

        if (!que_.empty()) {
            T item = std::move(que_.front());
            que_.pop();
            not_full_.notify_one();
            return item;
        }
        else if (is_close_) {
            return;
        }
    }

    bool try_pop(T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!que_.empty()) {
            item = std::move(que_.front());
            not_full_.notify_one();
            return true;
        }

        return false;
    }

    /// @brief 无线程安全保障，多线程下无意义
    /// @return 
    T& front() {
        return que_.front();
    }
};