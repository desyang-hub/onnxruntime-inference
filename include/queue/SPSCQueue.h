// SPSCQueue.h
#pragma once
#include <atomic>
#include <vector>
#include <memory>

template<typename T>
class SPSCQueue {
private:
    std::vector<T> buffer_;
    size_t capacity_;
    alignas(64) std::atomic<size_t> head_{0};  // 消费者索引
    alignas(64) std::atomic<size_t> tail_{0};  // 生产者索引
    
public:
    explicit SPSCQueue(size_t capacity) 
        : buffer_(capacity), capacity_(capacity) {
        // 确保容量是2的幂，便于取模优化
        // 如果不是，自动调整
        if (capacity & (capacity - 1)) {
            // 不是2的幂，向上取整
            size_t new_capacity = 1;
            while (new_capacity < capacity) new_capacity <<= 1;
            buffer_.resize(new_capacity);
            capacity_ = new_capacity;
        }
    }
    
    bool push(const T& item) {
        // 获取当前tail
        size_t tail = tail_.load(std::memory_order_relaxed);
        size_t next_tail = (tail + 1) & (capacity_ - 1);  // 使用位运算取模
        
        // 检查队列是否满
        if (next_tail == head_.load(std::memory_order_acquire)) {  // ✅ 正确
            return false;  // 队列满
        }
        
        // 写入数据
        buffer_[tail] = item;
        
        // 更新tail（release语义，确保前面的写完成）
        tail_.store(next_tail, std::memory_order_release);  // ✅ 正确
        return true;
    }

    bool push(T&& item) {
        // 获取当前tail
        size_t tail = tail_.load(std::memory_order_relaxed);
        size_t next_tail = (tail + 1) & (capacity_ - 1);  // 使用位运算取模
        
        // 检查队列是否满
        if (next_tail == head_.load(std::memory_order_acquire)) {  // ✅ 正确
            return false;  // 队列满
        }
        
        // 写入数据
        buffer_[tail] = std::move(item);
        
        // 更新tail（release语义，确保前面的写完成）
        tail_.store(next_tail, std::memory_order_release);  // ✅ 正确
        return true;
    }
    
    bool pop(T& item) {
        // 获取当前head
        size_t head = head_.load(std::memory_order_relaxed);
        
        // 检查队列是否空
        if (head == tail_.load(std::memory_order_acquire)) {  // ✅ 正确
            return false;  // 队列空
        }
        
        // 读取数据
        item = buffer_[head];
        
        // 更新head（release语义，确保前面的读完成）
        head_.store((head + 1) & (capacity_ - 1), std::memory_order_release);  // ✅ 正确
        return true;
    }
    
    bool empty() const {
        return head_.load(std::memory_order_acquire) == 
               tail_.load(std::memory_order_acquire);
    }
    
    size_t size() const {
        size_t head = head_.load(std::memory_order_acquire);
        size_t tail = tail_.load(std::memory_order_acquire);
        return (tail - head) & (capacity_ - 1);
    }
    
    size_t capacity() const {
        return capacity_ - 1;  // 实际可用容量
    }
};