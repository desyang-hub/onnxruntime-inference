/**
 * @FilePath     : /onnxruntime-infer/include/runner/ModelRunner.h
 * @Description  :  
 * @Author       : desyang
 * @Date         : 2026-07-01 14:13:01
 * @LastEditors  : desyang
 * @LastEditTime : 2026-07-01 20:35:21
**/
#pragma once

#include <memory>

#include "backend/InferenceBackend.h"

class ModelRunner
{
public:
    std::unique_ptr<InferenceBackend> backend_;
protected:
    explicit ModelRunner(std::unique_ptr<InferenceBackend> backend);
    
    ModelOutput infer();
};