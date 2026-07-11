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
#include <yaml-cpp/yaml.h>

#include "backend/InferenceBackend.h"

enum BackendType {
    BACKEND_ONNXRUNTIME,
    BACKEND_OPEN_VINO,
    BACKEND_OTHER_BACKEND
};

int ParseBackendTypeFromString(const std::string& backend);

class ModelRunner
{
protected:
    std::unique_ptr<InferenceBackend> backend_;

public:
    explicit ModelRunner(const YAML::Node& config);
    virtual ~ModelRunner() = default;
    
    ModelOutput infer(const TensorBuffer&);
};