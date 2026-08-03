#pragma once

#include <cuda_runtime.h>
#include "macor.h"
#include "exceptions/utils.h"

#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include "macor.h"
#include "exceptions/utils.h"

template<class T>
class CudaMallocGuard {
private:
    T* gpu_ptr_;

    // 仅用于非析构路径的带检查释放
    void checked_release() {
        if (gpu_ptr_) {
            CUDA_CHECK(cudaFree(gpu_ptr_));
            gpu_ptr_ = nullptr;
        }
    }

public:
    explicit CudaMallocGuard(size_t elements_size) : gpu_ptr_(nullptr) {
        allocate(elements_size);
    }

    CudaMallocGuard() noexcept : gpu_ptr_(nullptr) {}

    ~CudaMallocGuard() {
        //  析构绝不抛异常；cudaFree(nullptr) 是安全的，但这里仍判空避免多余调用
        if (gpu_ptr_) {
            cudaFree(gpu_ptr_);
        }
    }

    CudaMallocGuard(const CudaMallocGuard&) = delete;
    CudaMallocGuard& operator=(const CudaMallocGuard&) = delete;

    CudaMallocGuard(CudaMallocGuard&& other) noexcept
        : gpu_ptr_(other.gpu_ptr_) {
        other.gpu_ptr_ = nullptr;
    }

    CudaMallocGuard& operator=(CudaMallocGuard&& other) noexcept {
        if (this != &other) {
            //  移动赋值走 checked_release，因为这是显式用户操作
            checked_release();
            gpu_ptr_ = other.gpu_ptr_;
            other.gpu_ptr_ = nullptr;
        }
        return *this;
    }

    void allocate(size_t elements_size) {
        if (gpu_ptr_) {
            throw std::runtime_error(MESSAGE_WITH_LOC("CudaMallocGuard: already allocated"));
        }
        //  溢出检查
        if (elements_size > SIZE_MAX / sizeof(T)) {
            throw std::overflow_error(MESSAGE_WITH_LOC("CudaMallocGuard: allocation size overflow"));
        }
        CUDA_CHECK(cudaMalloc(&gpu_ptr_, sizeof(T) * elements_size));
    }

    //  移除不安全的模板 get，提供类型安全的 const/non-const 版本
    T*       get()       noexcept { return gpu_ptr_; }
    const T* get() const noexcept { return gpu_ptr_; }

    //  手动释放（带错误检查）
    void release() { checked_release(); }

    explicit operator bool() const noexcept { return gpu_ptr_ != nullptr; }
};