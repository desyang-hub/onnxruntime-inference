#pragma once

#include <atomic>
#include <chrono>
#include <thread>
#include <memory>

#include "BufferState.h"
#include "noncopyable.h"
#include "device/cuda_utils.h"


struct CPUBuffer : public noncopyable 
{
    std::unique_ptr<char[]> data_;
    std::atomic<BufferState> state{BufferState::IDLE};
    

    bool isAvailable(BufferState desired_state) {
        BufferState current = state.load();
        return current == BufferState::IDLE || 
               current == BufferState::COMPLETED;
    }

    CPUBuffer(size_t bytes_size) : data_(std::make_unique<char[]>(bytes_size)) {
    }

    template<class T = float>
    T* data() {
        return reinterpret_cast<T*>(data_.get());
    }
};