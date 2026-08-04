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

/// @brief 本质上和malloc没什么区别，但是初步猜测要想使用异步拷贝，似乎需要用这种方式申请内存，不过这个类经过封装，使用方式与普通指针已无差异。
/// @tparam T 存储的元素类型
template<class T>
class CudaMallocHostGuard {
private:
    T* host_ptr_;

    // 仅用于非析构路径的带检查释放
    void checked_release() {
        if (host_ptr_) {
            CUDA_CHECK(cudaFreeHost(host_ptr_));
            host_ptr_ = nullptr;
        }
    }

public:
    explicit CudaMallocHostGuard(size_t elements_size) : host_ptr_(nullptr) {
        allocate(elements_size);
    }

    CudaMallocHostGuard() noexcept : host_ptr_(nullptr) {}

    ~CudaMallocHostGuard() {
        //  析构绝不抛异常；cudaFree(nullptr) 是安全的，但这里仍判空避免多余调用
        if (host_ptr_) {
            cudaFreeHost(host_ptr_);
        }
    }

    CudaMallocHostGuard(const CudaMallocHostGuard&) = delete;
    CudaMallocHostGuard& operator=(const CudaMallocHostGuard&) = delete;

    CudaMallocHostGuard(CudaMallocHostGuard&& other) noexcept
        : host_ptr_(other.host_ptr_) {
        other.host_ptr_ = nullptr;
    }

    CudaMallocHostGuard& operator=(CudaMallocHostGuard&& other) noexcept {
        if (this != &other) {
            //  移动赋值走 checked_release，因为这是显式用户操作
            checked_release();
            host_ptr_ = other.host_ptr_;
            other.host_ptr_ = nullptr;
        }
        return *this;
    }


    T& operator[](int id) {
        return *(host_ptr_ + id);
    }

    const T& operator[](int id) const {
        return *(host_ptr_ + id);
    }

    void allocate(size_t elements_size) {
        if (host_ptr_) {
            throw std::runtime_error(MESSAGE_WITH_LOC("CudaMallocHostGuard: already allocated"));
        }
        //  溢出检查
        if (elements_size > SIZE_MAX / sizeof(T)) {
            throw std::overflow_error(MESSAGE_WITH_LOC("CudaMallocHostGuard: allocation size overflow"));
        }
        CUDA_CHECK(cudaMallocHost(&host_ptr_, sizeof(T) * elements_size));
    }

    //  移除不安全的模板 get，提供类型安全的 const/non-const 版本
    T*       get()       noexcept { return host_ptr_; }
    const T* get() const noexcept { return host_ptr_; }

    //  手动释放（带错误检查）
    void release() { checked_release(); }

    explicit operator bool() const noexcept { return host_ptr_ != nullptr; }
};