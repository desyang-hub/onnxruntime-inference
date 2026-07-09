/**
 * @FilePath     : /onnxruntime-infer/src/backend/InferenceBackend.cpp
 * @Description  :  
 * @Author       : desyang
 * @Date         : 2026-07-02 11:14:23
 * @LastEditors  : desyang
 * @LastEditTime : 2026-07-02 11:24:39
**/
#include "backend/InferenceBackend.h"

#include "logger/logger.h"

void InferenceBackend::init() {
    if (is_init_) return;
    tensorBuffer_ = TensorBuffer::create(shapes());
    is_init_ = true;
}

void InferenceBackend::warm_up(size_t cnt) {
    for (int i = 0; i < cnt; ++i) {
        run();
    }
    // std::cout << "热身已完成" << std::endl;
    LOG_INFO("Backend warm up successfully");
}