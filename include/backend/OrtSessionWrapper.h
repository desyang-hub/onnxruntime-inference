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
#include <unordered_map>

#include "Timer.h"
#include "arch/arch.h"
#include "TensorBuffer.h"
#include "InferenceBackend.h"
#include "device/cuda_utils.h"

GraphOptimizationLevel ParseGraphOptimizationLevel(const std::string& level);
ExecutionMode ParseExecutionMode(const std::string& mode);
OrtLoggingLevel ParseLogSeverityLevel(const std::string& level);

std::vector<int64_t> parse_input_meta(const std::vector<int64_t>& shape, const std::vector<int64_t>& img_shape);


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

    
    std::unique_ptr<Ort::Allocator> allocator_;
    Ort::Value input_tensor_{nullptr};
    float* pdata_;

    Ort::MemoryInfo active_mem_info_{nullptr};
    static constexpr size_t kGPUId = 0; // 使用gpu的设备编号

public:
    // gpu环境下返回GPU数据指针
    float* data() override {
        return pdata_;
    }

    explicit OrtSessionWrapper(const YAML::Node& config) : 
        env_(ORT_LOGGING_LEVEL_ERROR, "OrtSessionWrapper")
    {

        std::string model_path = config["path"].as<std::string>();
        // Ort::SessionOptions 配置
        Ort::SessionOptions session_options;

        // 从配置文件中获取sessionOptions配置
        std::vector<std::string> execution_providers = 
            config["session_options"]["execution_providers"]
            .as<std::vector<std::string>>(std::vector<std::string>{"CPUExecutionProvider"});
        std::string graph_optimization_level = 
            config["session_options"]["graph_optimization_level"]
            .as<std::string>("ORT_ENABLE_BASE");
        size_t intra_op_num_threads = 
            config["session_options"]["intra_op_num_threads"]
            .as<size_t>(1);
        size_t inter_op_num_threads = 
            config["session_options"]["inter_op_num_threads"]
            .as<size_t>(1);
        std::string execution_mode = 
            config["session_options"]["execution_mode"]
            .as<std::string>("ORT_SEQUENTIAL");
        std::string log_severity_level = 
            config["session_options"]["log_severity_level"]
            .as<std::string>("WARNING");
        bool enable_profiling = 
            config["session_options"]["enable_profiling"]
            .as<bool>(false);
        std::string optimized_model_path = 
            config["session_options"]["optimized_model_path"]
            .as<std::string>("");
        std::vector<int64_t> img_shape = 
            config["shape"]
            .as<std::vector<int64_t>>(std::vector<int64_t>{640, 640});
        size_t batch = config["batch"].as<size_t>(1);

        // 设置sessionOptions
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

        // TensorRT开启会异常，优化算子会推理异常
        auto set_optimized_model_path = [&optimized_model_path, &model_path]{
            // 如果优化模型路径能够被加载，直接加载优化路径
            if (fs::exists(optimized_model_path)) {
                model_path = optimized_model_path;
            } else if (!fs::exists(model_path)) {
                throw std::runtime_error("Model not found: " + model_path);
            }
        };


        // 获取环境中可用的推理设备
        auto execute_providers = Ort::GetAvailableProviders();
        // for (int i = 0; i < execute_providers.size(); ++i) {
        //     std::cout << execute_providers[i] << std::endl;
        // }

#ifdef ENABLE_CUDA
        static constexpr unsigned long long kMaxGPUMemLimit = 4; // 限制4GB显存空间

        // 1. 尝试 TensorRT
        const std::string& tensorrt_provider = "TensorrtExecutionProvider";
        bool isTensorRTAvailable = std::find(execute_providers.begin(), 
                                             execute_providers.end(), 
                                             tensorrt_provider) != execute_providers.end();

        if (isTensorRTAvailable) {
            try
            {
                OrtTensorRTProviderOptions trt_opts{};
                trt_opts.device_id = kGPUId;
                trt_opts.trt_max_workspace_size = kMaxGPUMemLimit * 1024 * 1024 * 1024;
                trt_opts.trt_engine_cache_enable = 1;
                trt_opts.trt_engine_cache_path = "./trt_cache";  // 关键：指定缓存目录
                session_options.AppendExecutionProvider_TensorRT(trt_opts);
                std::cout << "[EP] TensorRT registered" << std::endl;
            } catch(const std::exception& e) {
                std::cerr << "[EP] TensorRT unavailable: " << e.what() << std::endl;
                set_optimized_model_path();
            }
        } else {
            set_optimized_model_path();
        }

        // 添加CUDA设备
        const std::string& cuda_provider = "CUDAExecutionProvider";
        bool isCudaAvailable = std::find(execute_providers.begin(), 
                                         execute_providers.end(), 
                                         cuda_provider) != execute_providers.end();

        // 目标设备可用
        if (isCudaAvailable) {    
            try
            {
                // 设置CUDA选项
                OrtCUDAProviderOptions cuda_options{};
                cuda_options.device_id = kGPUId;
                cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic; // 快速启动
                cuda_options.gpu_mem_limit = kMaxGPUMemLimit * 1024 * 1024 * 1024;               // 限制 4GB
                cuda_options.do_copy_in_default_stream = 1;
                session_options.AppendExecutionProvider_CUDA(cuda_options);
                std::cout << "[EP] CUDA registered" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[EP] CUDA unavailable: " << e.what() << std::endl;
            }
        }
#endif

        try
        {   // OrtSession 初始化
            session_ = std::make_unique<Ort::Session>(env_, MODEL_PATH(model_path).c_str(), session_options);
        }
        catch(const std::exception& e)
        {
            throw std::runtime_error("OrtSession create failed!");
        }

        // 获取当前推理设备信息
        ResolveActiveDevice();

        // 获取模型信息
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


        std::vector<int64_t> raw_input_shapes = 
            session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

        // batch 设置
        input_shapes_ = parse_input_meta(raw_input_shapes, img_shape);
        input_shapes_[0] = batch; // batch
        output_shapes_ = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

        // 进行张量初始化
        input_tensor_init();

        // 进行热身
        size_t warm_up_num = config["warm_up"].as<size_t>(0);
        warm_up(warm_up_num);
    }

    // ✅ 从已创建的 Session 中反查真实设备
    void ResolveActiveDevice() {
        // 按优先级从高到低探测
        struct DeviceCandidate {
            const char* name;
            int device_id;
        };
        
        // TODO: 此处硬编码，待解决
        DeviceCandidate candidates[] = {
            {"Tensorrt", kGPUId},
            {"Cuda", kGPUId},
        };
        
        for (const auto& candidate : candidates) {
            try {
                Ort::MemoryInfo mem_info{candidate.name, OrtDeviceAllocator, 
                                         candidate.device_id, OrtMemTypeDefault};
                // 关键：用 Session 构造 Allocator
                // 如果该设备未被 Session 接受，这里会抛异常
                allocator_ = std::make_unique<Ort::Allocator>(*session_, mem_info);
                Ort::Allocator allocator(*session_, mem_info);
                
                setactivateGPUId(candidate.device_id);
                enableGPUActivate();
                std::cout << "[Device] ✅ Active EP: " << candidate.name 
                          << ", logical device " << activateGPUId() << std::endl;
                return; // 找到最高优先级的可用设备，立即返回
            } catch (std::exception& e) {
                // 该设备不可用，继续尝试下一个
                std::cout << "[Device] " << candidate.name 
                          << " not active, trying next... err: " << e.what() << std::endl;
            }
        }
        
        // 所有 GPU 候选都失败
        std::cout << "[Device] ⚠️  All GPU EPs unavailable, fallback to CPU" << std::endl;
    }

    const std::vector<int64_t>& shapes() const override {
        return input_shapes_;
    }

    // 初始化张量，使得后续的推理全部使用该张量上进行
    void input_tensor_init() {
        // 如果GPU可用
        if (isGPUActivate()) {
            // 缓存推理的数据张量
            input_tensor_ = Ort::Value::CreateTensor<float>(
                *allocator_,
                input_shapes_.data(),
                input_shapes_.size()
            );

            pdata_ = input_tensor_.GetTensorMutableData<float>();
        } else {
            auto& tensor_buffer = tensorBuffer();
            Ort::MemoryInfo ort_memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
            input_tensor_ = Ort::Value::CreateTensor<float>(
                ort_memory_info,
                (float*)tensor_buffer.data,
                tensor_buffer.num_elements,
                tensor_buffer.shape.data(),
                tensor_buffer.shape.size()
            );

            pdata_ = input_tensor_.GetTensorMutableData<float>();
        }
    }

    ModelOutput run() override {
        static constexpr size_t kInputCount = 1; // 这个是输入节点的数量，不是batch，多模态模型才会有多输入

        auto& tensor_buffer = tensorBuffer();
        if (!tensor_buffer.valid()) {
            throw std::runtime_error("Data not valid");
        }

#ifdef ENABLE_CUDA
        // 使用异步stream进行数据拷贝时，使用内存屏障来保证GPU数据同步
        cudaDeviceSynchronize();
#endif

        // 进行推理
        TIMER_START_TAG(Run)
        std::vector<Ort::Value> ort_outputs = session_->Run(
            Ort::RunOptions{},
            input_names_c_str_.data(),
            &input_tensor_,
            input_names_c_str_.size(), // 单个节点
            output_names_c_str_.data(),
            output_names_c_str_.size()
        );
        TIMER_FINISH_TAG(Run);


        // ========== 3. Ort::Value → TensorBuffer ==========
        ModelOutput result;
        for (size_t i = 0; i < ort_outputs.size(); ++i) {
            auto& val = ort_outputs[i];
            auto type_info = val.GetTensorTypeAndShapeInfo();
            
            // 提取 shape
            std::vector<int64_t> shape = type_info.GetShape();
            
            // 零拷贝包装：不复制数据，但绑定 Ort::Value 的生命周期
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

            // 此处赋值不合理
            result.tensors[output_names_[i]].letterbox_params = tensor_buffer.letterbox_params;
        }

        return result;
    }
};