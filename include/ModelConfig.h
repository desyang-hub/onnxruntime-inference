/**
 * @FilePath     : /onnxruntime-infer/include/ModelConfig.h
 * @Description  :  
 * @Author       : desyang
 * @Date         : 2026-06-30 20:01:13
 * @LastEditors  : desyang
 * @LastEditTime : 2026-06-30 20:01:15
**/
#pragma once

#include <string>

struct ModelConfig
{
    std::string model_path;
    bool warm_up;
};
