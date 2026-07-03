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
    throw "Unknow graph optimization level: " + level;
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