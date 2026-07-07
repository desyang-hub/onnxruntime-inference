/**
 * @FilePath     : /onnxruntime-infer/src/backend/OrtSessionWrapper.cpp
 * @Description  :  
 * @Author       : desyang
 * @Date         : 2026-07-03 10:48:43
 * @LastEditors  : desyang
 * @LastEditTime : 2026-07-03 10:49:58
**/
#include "backend/OrtSessionWrapper.h"

// ==================== GraphOptimizationLevel ====================
static const std::unordered_map<std::string, GraphOptimizationLevel> kOptLevelMap = {
    {"ORT_DISABLE_ALL",  ORT_DISABLE_ALL},
    {"ORT_ENABLE_BASIC", ORT_ENABLE_BASIC},   // 注意：旧版本叫 BASIC
    {"ORT_ENABLE_EXTENDED", ORT_ENABLE_EXTENDED},
    {"ORT_ENABLE_ALL",   ORT_ENABLE_ALL},
};

GraphOptimizationLevel ParseGraphOptimizationLevel(const std::string& level) {
    auto it = kOptLevelMap.find(level);
    if (it != kOptLevelMap.end()) {
        return it->second;
    }
    throw std::runtime_error("Unknow graph optimization level: " + level);
}

// ==================== ExecutionMode ====================
static const std::unordered_map<std::string, ExecutionMode> kExecutionModeMap = {
    {"ORT_SEQUENTIAL", ORT_SEQUENTIAL},
    {"ORT_PARALLEL",   ORT_PARALLEL},
};

ExecutionMode ParseExecutionMode(const std::string& mode) {
    auto it = kExecutionModeMap.find(mode);
    if (it != kExecutionModeMap.end()) {
        return it->second;
    }
    throw std::invalid_argument("Unknown execution mode: " + mode);
}

// ==================== LogSeverityLevel ====================
static const std::unordered_map<std::string, OrtLoggingLevel> kLogSeverityMap = {
    {"VERBOSE",  ORT_LOGGING_LEVEL_VERBOSE},
    {"INFO",     ORT_LOGGING_LEVEL_INFO},
    {"WARNING",  ORT_LOGGING_LEVEL_WARNING},
    {"ERROR",    ORT_LOGGING_LEVEL_ERROR},
    {"FATAL",    ORT_LOGGING_LEVEL_FATAL},
};

OrtLoggingLevel ParseLogSeverityLevel(const std::string& level) {
    auto it = kLogSeverityMap.find(level);
    if (it != kLogSeverityMap.end()) {
        return it->second;
    }
    throw std::invalid_argument("Unknown log severity level: " + level);
}


// (b, c, h, w)
std::vector<int64_t> parse_input_meta(const std::vector<int64_t>& shape, const std::vector<int64_t>& img_shape) {
    if (shape.size() != 4) {
        throw std::runtime_error("Expected 4D input, got " 
                                 + std::to_string(shape.size()) + "D");
    }

    std::vector<int64_t> infer_shape(shape.size());

    // ⭐ 核心逻辑：channels 一定是 1、3 或 4
    //    而 H/W 通常 >= 224，绝不可能 <= 4
    if (shape[1] == 1 || shape[1] == 3 || shape[1] == 4) {
        // NCHW: [N, C, H, W]
        infer_shape = shape;
    } else if (shape[3] == 1 || shape[3] == 3 || shape[3] == 4) {
        // NHWC: [N, H, W, C]
        infer_shape = {shape[0], shape[3], shape[1], shape[2]};
    } else {
        throw std::runtime_error(
            "Cannot determine layout: no dimension equals 1/3/4. "
            "Shape: [" + std::to_string(shape[0]) + ", " 
                        + std::to_string(shape[1]) + ", " 
                        + std::to_string(shape[2]) + ", " 
                        + std::to_string(shape[3]) + "]");
    }

    // 动态维度 (-1, 3, -1, -1)
    if (infer_shape[2]) {
        infer_shape[2] = img_shape[0]; // height
        infer_shape[3] = img_shape[1]; // width
    }

    return infer_shape;
}