/**
 * @FilePath     : /onnxruntime-infer/include/OnnxInfer.h
 * @Description  :  
 * @Author       : desyang
 * @Date         : 2026-06-30 19:50:04
 * @LastEditors  : desyang
 * @LastEditTime : 2026-07-01 21:02:51
**/
#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <iostream>

#include "ModelConfig.h"

class OnnxInfer
{
    struct ort_session_deleter {
        void operator()(Ort::Session* session) {
            session->release();
            delete session;
        }
    };

protected:
    Ort::Env env_; // Ort推理环境
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session, ort_session_deleter> session_;

    std::vector<std::string> input_node_names_;
    std::vector<std::string> output_node_names_;
    std::vector<int64_t> inputShapes_;
    std::vector<int64_t> outputShapes_;
    std::vector<const char*> inputNames_;
    std::vector<const char*> outputNames_;

    // 模型热身
    void dummy_inference() {
        std::cout << "warm up start " << std::endl;
        size_t dummy_input_size = 
        inputShapes_[CHANNELS] * inputShapes_[WIDTH] * inputShapes_[HEIGHT];
        // 准备数据
        std::vector<float> dummy_input_data(dummy_input_size, 0.0f);
        std::vector<int64_t> dummy_input_shape{1, inputShapes_[CHANNELS], inputShapes_[HEIGHT], inputShapes_[WIDTH]};

        Ort::MemoryInfo dummy_memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        Ort::Value dummy_input_tensor = Ort::Value::CreateTensor<float>(
            dummy_memory_info,
            dummy_input_data.data(),
            dummy_input_data.size(),
            dummy_input_shape.data(),
            dummy_input_shape.size()
        );

        for (int i = 0; i < 3; ++i) {
            session_->Run(
                Ort::RunOptions{},
                inputNames_.data(),
                &dummy_input_tensor,
                1,
                outputNames_.data(),
                outputNames_.size()
            );
        }

        std::cout << "warm up success" << std::endl;
    }

public:
    enum ShapeId {
        BATCH,
        CHANNELS,
        WIDTH,
        HEIGHT
    };

    /**
     * @brief 构造函数，初始化ONNX推理引擎
     * @param config 模型配置，包含模型路径等参数
     */
    explicit OnnxInfer(const ModelConfig& config) : env_(ORT_LOGGING_LEVEL_ERROR, "OnnxInfer") {
        // 设置图优化级别和线程数
        // 所有的模型的配置load应该是一样的；
        session_options_.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);  // 启用基础图优化
        session_options_.SetInterOpNumThreads(1);  // 设置操作间线程数为1

        
        // 创建ONNX会话
        std::unique_ptr<Ort::Session, ort_session_deleter> 
        session(new Ort::Session(env_, config.model_path.c_str(), session_options_));
        // 创建推理会话
        session_ = std::move(session);


        // 输入输出节点的个数
        size_t input_nodes_num = session_->GetInputCount();
        size_t output_nodes_num = session_->GetOutputCount();
        input_node_names_.reserve(input_nodes_num);
        output_node_names_.reserve(output_nodes_num);

        Ort::AllocatorWithDefaultOptions ort_allocator;
        inputShapes_ = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        for (size_t i = 0; i < input_nodes_num; ++i) {
            input_node_names_.push_back(session_->GetInputNameAllocated(i, ort_allocator).get());
        }

        outputShapes_ = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        for (size_t i = 0; i < output_nodes_num; ++i) {
            output_node_names_.push_back(session_->GetOutputNameAllocated(i, ort_allocator).get());
        }

        inputNames_ = std::vector<const char*>{input_node_names_[0].c_str()};
        outputNames_ = std::vector<const char*>{output_node_names_[0].c_str()};


        if (config.warm_up) dummy_inference();
    }

    ~OnnxInfer() {
        session_options_.release();
    }


    virtual void run(const cv::Mat& img) {
        
    }

    std::vector<Ort::Value> inference(const cv::Mat& blob) {
        Ort::MemoryInfo ort_memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        std::vector<int64_t> input_shape_info{1, inputShapes_[CHANNELS], inputShapes_[HEIGHT], inputShapes_[WIDTH]};

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            ort_memory_info,
            (float*)blob.ptr<float>(),
            blob.total(),
            input_shape_info.data(),
            input_shape_info.size()
        );

        return session_->Run(
            Ort::RunOptions{},
            inputNames_.data(),
            &input_tensor,
            1,
            outputNames_.data(),
            outputNames_.size()
        );
    }

    void infer(const cv::Mat& img) {
        // 预处理阶段
        Ort::Value input_tensor = preprocess(img);

        // onnxruntime 推理阶段
        std::vector<Ort::Value> output_tensor = session_->Run(
            Ort::RunOptions{},
            inputNames_.data(),
            &input_tensor,
            1,
            outputNames_.data(),
            outputNames_.size()
        );

        // 后处理阶段
        postprocess(output_tensor);
    }

    // 数据预处理
    virtual Ort::Value preprocess(const cv::Mat& img) const {
        return Ort::Value{ nullptr };
    }

    // 数据后处理
    virtual void postprocess(const std::vector<Ort::Value>& output_tensor) const {}
};