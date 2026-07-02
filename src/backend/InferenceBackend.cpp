/**
 * @FilePath     : /onnxruntime-infer/src/backend/InferenceBackend.cpp
 * @Description  :  
 * @Author       : desyang
 * @Date         : 2026-07-02 11:14:23
 * @LastEditors  : desyang
 * @LastEditTime : 2026-07-02 11:24:39
**/
#include "backend/InferenceBackend.h"


void InferenceBackend::warm_up(size_t cnt) {
    TensorBuffer tensor_buf = TensorBuffer::create(shapes());
    for (int i = 0; i < cnt; ++i) {
        run(tensor_buf);
    }
}