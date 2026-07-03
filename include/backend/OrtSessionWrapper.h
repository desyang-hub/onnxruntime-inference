/**
 * @FilePath     : /onnxruntime-infer/include/backend/OrtSessionWrapper.h
 * @Description  :  
 * @Author       : desyang
 * @Date         : 2026-07-01 13:28:35
 * @LastEditors  : desyang
 * @LastEditTime : 2026-07-03 11:05:03
**/
#pragma once

#include <string>
#include <filesystem>
#include <stdexcept>
#include <yaml-cpp/yaml.h>
#include <onnxruntime_cxx_api.h>

#include "arch/arch.h"
#include "InferenceBackend.h"

GraphOptimizationLevel ParseGraphOptimizationLevel(const std::string& level);
ExecutionMode ParseExecutionMode(const std::string& mode);
OrtLoggingLevel ParseLogSeverityLevel(const std::string& level);


class OrtSessionWrapper : public InferenceBackend {
    enum ShapeId {
        BATCH,
        CHANNELS,
        HEIGHT,
        WIDTH
    };

private:
    Ort::Env env_; // Ort推理环境
    // Ort::Session session_;
    std::unique_ptr<Ort::Session> session_;

    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<int64_t> input_shapes_;
    std::vector<int64_t> output_shapes_;
    std::vector<const char*> input_names_c_str_;
    std::vector<const char*> output_names_c_str_;


    // (b, c, h, w)
    std::vector<int64_t> parse_input_meta(const std::vector<int64_t>& shape) {
        if (shape.size() != 4) {
            throw std::runtime_error("Expected 4D input, got " 
                                     + std::to_string(shape.size()) + "D");
        }
    
        // ⭐ 核心逻辑：channels 一定是 1、3 或 4
        //    而 H/W 通常 >= 224，绝不可能 <= 4
        if (shape[1] == 1 || shape[1] == 3 || shape[1] == 4) {
            // NCHW: [N, C, H, W]
            return shape;
        } else if (shape[3] == 1 || shape[3] == 3 || shape[3] == 4) {
            // NHWC: [N, H, W, C]
            return {shape[0], shape[3], shape[1], shape[2]};
        } else {
            throw std::runtime_error(
                "Cannot determine layout: no dimension equals 1/3/4. "
                "Shape: [" + std::to_string(shape[0]) + ", " 
                            + std::to_string(shape[1]) + ", " 
                            + std::to_string(shape[2]) + ", " 
                            + std::to_string(shape[3]) + "]");
        }
    }


public:
    explicit OrtSessionWrapper(const YAML::Node& config) : 
        env_(ORT_LOGGING_LEVEL_ERROR, "OrtSessionWrapper")
    {

        std::string model_path = config["path"].as<std::string>();

        // Ort::SessionOptions 配置
        Ort::SessionOptions session_options;

        // execution_providers: ["CPUExecutionProvider"]
        // graph_optimization_level: "ORT_ENABLE_ALL"
        // intra_op_num_threads: 4
        // inter_op_num_threads: 1
        // execution_mode: "SEQUENTIAL"
        // log_severity_level: "WARNING"
        // enable_profiling: false,
        // optimized_model_path: ""

        std::vector<std::string> execution_providers = 
        config["session_options"]["execution_providers"].as<std::vector<std::string>>(std::vector<std::string>{"CPUExecutionProvider"});
        std::string graph_optimization_level = config["session_options"]["graph_optimization_level"].as<std::string>("ORT_ENABLE_BASE");
        size_t intra_op_num_threads = config["session_options"]["intra_op_num_threads"].as<size_t>(1);
        size_t inter_op_num_threads = config["session_options"]["inter_op_num_threads"].as<size_t>(1);
        std::string execution_mode = config["session_options"]["execution_mode"].as<std::string>("ORT_SEQUENTIAL");
        std::string log_severity_level = config["session_options"]["log_severity_level"].as<std::string>("WARNING");
        bool enable_profiling = config["session_options"]["enable_profiling"].as<bool>(false);
        std::string optimized_model_path = config["session_options"]["optimized_model_path"].as<std::string>("");

        static const std::unordered_map<std::string, GraphOptimizationLevel> kOptLevelMap = {
            {"ORT_DISABLE_ALL",  ORT_DISABLE_ALL},
            {"ORT_ENABLE_BASIC", ORT_ENABLE_BASIC},   // 注意：旧版本叫 BASIC
            {"ORT_ENABLE_EXTENDED", ORT_ENABLE_EXTENDED},
            {"ORT_ENABLE_ALL",   ORT_ENABLE_ALL},
        };

        session_options.SetGraphOptimizationLevel(ParseGraphOptimizationLevel(graph_optimization_level));
        session_options.SetIntraOpNumThreads(intra_op_num_threads);
        session_options.SetInterOpNumThreads(inter_op_num_threads);
        session_options.SetExecutionMode(ParseExecutionMode(execution_mode));
        session_options.SetLogSeverityLevel(ParseLogSeverityLevel(log_severity_level));

        if (enable_profiling) {
            session_options.EnableProfiling("profiling");
        }

        namespace fs = std::filesystem;
        if (optimized_model_path.empty()) {
            // 配置路径
            fs::path p(model_path);
            // 提取文件名（不含扩展名）+ 新前缀 + 原扩展名
            fs::path new_path = p.parent_path() / ("optimized_" + p.stem().string() + p.extension().string());
            optimized_model_path = new_path.string();
        }

        if (!fs::exists(optimized_model_path)) {
            session_options.SetOptimizedModelFilePath(optimized_model_path.c_str());
        }

        // 如果优化模型路径能够被加载，直接加载优化路径
        if (fs::exists(optimized_model_path)) {
            model_path = optimized_model_path;
        } else if (!fs::exists(model_path)) {
            throw "Model not found: " + model_path;
        }
        
        session_ = std::make_unique<Ort::Session>(env_, MODEL_PATH(model_path).c_str(), session_options);

        // 初始化的过程
        size_t input_names_num = session_->GetInputCount();
        size_t output_names_num = session_->GetOutputCount();

        input_names_.reserve(input_names_num);
        output_names_.reserve(output_names_num);

        Ort::AllocatorWithDefaultOptions ort_allocator;
        for (int i = 0; i < input_names_num; ++i) {
            input_names_.push_back(session_->GetInputNameAllocated(i, ort_allocator).get());
        }

        for (int i = 0; i < output_names_num; ++i) {
            output_names_.push_back(session_->GetOutputNameAllocated(i, ort_allocator).get());
        }

        input_names_c_str_.push_back(input_names_[0].c_str());
        output_names_c_str_.push_back(output_names_[0].c_str());


        std::vector<int64_t> raw_input_shapes = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        input_shapes_ = parse_input_meta(raw_input_shapes);
        output_shapes_ = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    }

    const std::vector<int64_t>& shapes() const override {
        return input_shapes_;
    }

    ModelOutput run(const TensorBuffer& tensor_buffer) override {
        // 真正的 ORT 调用在这里
        if (!tensor_buffer.valid()) {
            throw std::runtime_error("Data not valid");
        }
        
        Ort::MemoryInfo ort_memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        // 创建推理的数据张量
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            ort_memory_info,
            (float*)tensor_buffer.data,
            tensor_buffer.byte_size(),
            tensor_buffer.shape.data(),
            tensor_buffer.shape.size()
        );

        // 进行推理
        std::vector<Ort::Value> ort_outputs = session_->Run(
            Ort::RunOptions{},
            input_names_c_str_.data(),
            &input_tensor,
            tensor_buffer.shape[0], // batch
            output_names_c_str_.data(),
            output_names_c_str_.size()
        );


        // ========== 3. ⭐ 核心翻译：Ort::Value → TensorBuffer ==========
        ModelOutput result;
        for (size_t i = 0; i < ort_outputs.size(); ++i) {
            auto& val = ort_outputs[i];
            auto type_info = val.GetTensorTypeAndShapeInfo();
            
            // 提取 shape
            std::vector<int64_t> shape = type_info.GetShape();
            
            // ⭐ 零拷贝包装：不复制数据，但绑定 Ort::Value 的生命周期
            // ort_outputs 会被 move 到 lambda 捕获中，保证指针有效
            float* data_ptr = val.GetTensorMutableData<float>();
            
            // 创建生命周期守卫，防止 Ort::Value 提前析构
            auto guard = std::shared_ptr<Ort::Value>(
                new Ort::Value(std::move(val)),
                [](Ort::Value* p) { delete p; }
            );
            
            result.tensors[output_names_[i]] = TensorBuffer::wrap(
                data_ptr, shape, 
                std::reinterpret_pointer_cast<void>(guard)
            );
        }

        return result;

    }
};