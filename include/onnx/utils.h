#pragma once

#include <onnxruntime_cxx_api.h>

GraphOptimizationLevel ParseGraphOptimizationLevel(const std::string &level);
ExecutionMode ParseExecutionMode(const std::string &mode);
OrtLoggingLevel ParseLogSeverityLevel(const std::string &level);

std::vector<int64_t> parse_input_meta(const std::vector<int64_t> &shape, const std::vector<int64_t> &img_shape);