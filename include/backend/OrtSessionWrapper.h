/**
 * @FilePath     : /onnxruntime-infer/include/backend/OrtSessionWrapper.h
 * @Description  :  
 * @Author       : desyang
 * @Date         : 2026-07-01 13:28:35
 * @LastEditors  : desyang
 * @LastEditTime : 2026-07-01 21:47:30
**/
#pragma once

#include <string>
#include <stdexcept>
#include <onnxruntime_cxx_api.h>

#include "arch/arch.h"
#include "InferenceBackend.h"

class OrtSessionWrapper : public InferenceBackend {
    enum ShapeId {
        BATCH,
        CHANNELS,
        HEIGHT,
        WIDTH
    };

private:
    Ort::Env env_; // Ort推理环境
    Ort::Session session_;

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
    explicit OrtSessionWrapper(const std::string& model_path, const Ort::SessionOptions& options = {}) : env_(ORT_LOGGING_LEVEL_ERROR, "OrtSessionWrapper"), session_(env_, MODEL_PATH(model_path).c_str(), options) {
        // 初始化的过程
        size_t input_names_num = session_.GetInputCount();
        size_t output_names_num = session_.GetOutputCount();

        input_names_.reserve(input_names_num);
        output_names_.reserve(output_names_num);

        Ort::AllocatorWithDefaultOptions ort_allocator;
        for (int i = 0; i < input_names_num; ++i) {
            input_names_.push_back(session_.GetInputNameAllocated(i, ort_allocator).get());
        }

        for (int i = 0; i < output_names_num; ++i) {
            output_names_.push_back(session_.GetOutputNameAllocated(i, ort_allocator).get());
        }

        input_names_c_str_.push_back(input_names_[0].c_str());
        output_names_c_str_.push_back(output_names_[0].c_str());


        std::vector<int64_t> raw_input_shapes = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        input_shapes_ = parse_input_meta(raw_input_shapes);
        output_shapes_ = session_.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    }

    ~OrtSessionWrapper() {
        session_.release();
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
        std::vector<Ort::Value> ort_outputs = session_.Run(
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