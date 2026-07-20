#include "BufferPool.h"


BufferPool::BufferPool(int buffer_size, int element_bytes_size) : is_close_(false) {

    total_bytes_ = static_cast<size_t>(buffer_size) * element_bytes_size;

    data_ = std::unique_ptr<char[]>(new char[total_bytes_]);

    for (int i = 0; i < buffer_size; ++i) {
        que_.push(data_.get() + i * element_bytes_size);
    }
}

BufferPool::~BufferPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        is_close_ = true;
    }
    condition_.notify_all();
}

