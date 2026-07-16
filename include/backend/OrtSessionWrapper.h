/**
 * @FilePath     : /onnxruntime-inference/include/backend/OrtSessionWrapper.h
 * @Description  :
 * @Author       : desyang
 * @Date         : 2026-07-01 13:28:35
 * @LastEditors  : desyang
 * @LastEditTime : 2026-07-07 17:25:18
 **/
#pragma once

#include <string>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <yaml-cpp/yaml.h>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>
#include <unordered_map>

#include "Timer.h"
#include "arch/arch.h"
#include "TensorBuffer.h"
#include "InferenceBackend.h"
#include "device/cuda_utils.h"
#include "logger/logger.h"
#include "ScopedTimer.h"

GraphOptimizationLevel ParseGraphOptimizationLevel(const std::string &level);
ExecutionMode ParseExecutionMode(const std::string &mode);
OrtLoggingLevel ParseLogSeverityLevel(const std::string &level);

std::vector<int64_t> parse_input_meta(const std::vector<int64_t> &shape, const std::vector<int64_t> &img_shape);

/// @brief OnnxRuntime 推理器
class OrtSessionWrapper : public InferenceBackend
{
    enum ShapeId
    {
        BATCH,
        CHANNELS,
        HEIGHT,
        WIDTH
    };

private:
    Ort::Env env_; // Ort推理环境
    std::unique_ptr<Ort::Session> session_;

    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<int64_t> input_shapes_;
    std::vector<int64_t> output_shapes_;
    std::vector<const char *> input_names_c_str_;
    std::vector<const char *> output_names_c_str_;

    std::unique_ptr<Ort::Allocator> allocator_;
    Ort::Value input_tensor_{nullptr};
    float *pdata_;

    Ort::MemoryInfo active_mem_info_{nullptr};
    int kGPUId = -1; // 使用gpu的设备编号

    std::unordered_map<float*, Ort::Value> input_tensors_;

public:
    // gpu环境下返回GPU数据指针
    float *data() override
    {
        return pdata_;
    }

    explicit OrtSessionWrapper(const YAML::Node &config) : env_(ORT_LOGGING_LEVEL_ERROR, "OrtSessionWrapper")
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
        int intra_op_num_threads =
            config["session_options"]["intra_op_num_threads"]
                .as<int>(1);
        int inter_op_num_threads =
            config["session_options"]["inter_op_num_threads"]
                .as<int>(1);
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
        kGPUId = config["gpu"].as<int>(-1);

        // 设置sessionOptions
        session_options.SetGraphOptimizationLevel(ParseGraphOptimizationLevel(graph_optimization_level));
        session_options.SetIntraOpNumThreads(intra_op_num_threads);
        session_options.SetInterOpNumThreads(inter_op_num_threads);
        session_options.SetExecutionMode(ParseExecutionMode(execution_mode));
        session_options.SetLogSeverityLevel(ParseLogSeverityLevel(log_severity_level));

        if (enable_profiling)
        {
            session_options.EnableProfiling(STRING_TO_WSTRING("profiling").c_str());
        }

        namespace fs = std::filesystem;
        if (optimized_model_path.empty())
        {
            // 配置路径
            fs::path p(model_path);
            // 提取文件名（不含扩展名）+ 新前缀 + 原扩展名
            fs::path new_path = p.parent_path() / ("optimized_" + p.stem().string() + p.extension().string());
            optimized_model_path = new_path.string();
        }


        // 获取环境中可用的推理设备
        // auto execute_providers = Ort::GetAvailableProviders();
        std::vector<std::string> execute_providers =
            config["session_options"]["execution_providers"]
                .as<std::vector<std::string>>();
        // for (int i = 0; i < execute_providers.size(); ++i) {
        //     std::cout << execute_providers[i] << std::endl;
        // }

        bool isTensorRTAvailable = false;

#ifdef ENABLE_CUDA
        static constexpr unsigned long long kMaxGPUMemLimit = 4; // 限制4GB显存空间

        // 1. 尝试 TensorRT
        // const std::string &tensorrt_provider = "TensorrtExecutionProvider";
        // isTensorRTAvailable = std::find(execute_providers.begin(),
        //                                      execute_providers.end(),
        //                                      tensorrt_provider) != execute_providers.end();

        // if (isTensorRTAvailable)
        // {
        //     try
        //     {
        //         OrtTensorRTProviderOptions trt_opts{};
        //         trt_opts.device_id = kGPUId;
        //         trt_opts.trt_max_workspace_size = kMaxGPUMemLimit * 1024 * 1024 * 1024;
        //         trt_opts.trt_engine_cache_enable = 1;
        //         trt_opts.trt_engine_cache_path = "./trt_cache"; // 关键：指定缓存目录
        //         session_options.AppendExecutionProvider_TensorRT(trt_opts);
        //         LOG_INFO("[EP] TensorRT registered");
        //     }
        //     catch (const std::exception &e)
        //     {
        //         LOG_WARN("[EP] TensorRT unavailable: {}", e.what());
        //         isTensorRTAvailable = false;
        //     }
        // }


        const std::string &tensorrt_provider = "TensorrtExecutionProvider";
        isTensorRTAvailable = std::find(execute_providers.begin(),
                                        execute_providers.end(),
                                        tensorrt_provider) != execute_providers.end();

        // 使用V2 api开启TensorRt
        if (isTensorRTAvailable)
        {
            try
            {
                // 1. 获取全局 C API 指针（所有版本都有这个函数）
                const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
                if (!ort_api) {
                    throw std::runtime_error("Failed to get ONNX Runtime C API");
                }
        
                // 2. 创建 V2 Options
                OrtTensorRTProviderOptionsV2* trt_opts = nullptr;
                OrtStatus* status = ort_api->CreateTensorRTProviderOptions(&trt_opts);
                if (status != nullptr) {
                    std::string err = ort_api->GetErrorMessage(status);
                    ort_api->ReleaseStatus(status);
                    throw std::runtime_error("CreateTensorRTProviderOptions failed: " + err);
                }
        
                // 3. 配置参数
                std::string workspace_size = std::to_string(kMaxGPUMemLimit * 1024ULL * 1024ULL * 1024ULL);
                std::string device_id_str  = std::to_string(kGPUId);
        
                const char* keys[] = {
                    "device_id",
                    "trt_max_workspace_size",
                    "trt_fp16_enable",
                    "trt_engine_cache_enable",
                    "trt_engine_cache_path"
                };

                std::string FP16_enable = config["session_options"]["trt_fp16_enable"].as<std::string>("0");

                const char* values[] = {
                    device_id_str.c_str(),
                    workspace_size.c_str(),
                    FP16_enable.c_str(),   // FP16 ON["1"], OFF["0"]
                    "1",   // Cache ON
                    "./trt_cache"
                };
        
                status = ort_api->UpdateTensorRTProviderOptions(
                    trt_opts, keys, values, sizeof(keys) / sizeof(keys[0]));
                if (status != nullptr) {
                    std::string err = ort_api->GetErrorMessage(status);
                    ort_api->ReleaseStatus(status);
                    ort_api->ReleaseTensorRTProviderOptions(trt_opts);
                    throw std::runtime_error("UpdateTensorRTProviderOptions failed: " + err);
                }
        
                // 4. 挂载到 C++ SessionOptions（C/C++ 混合调用关键步骤）
                session_options.AppendExecutionProvider_TensorRT_V2(*trt_opts);
        
                // 5. 释放 V2 Options
                ort_api->ReleaseTensorRTProviderOptions(trt_opts);
                
                if (FP16_enable == "1")
                    LOG_INFO("[EP] TensorRT registered (FP16 enabled via C API)");
                else
                    LOG_INFO("[EP] TensorRT registered (FP32 enabled via C API)");
            }
            catch (const std::exception &e)
            {
                LOG_WARN("[EP] TensorRT unavailable: {}", e.what());
                isTensorRTAvailable = false;
            }
        }


        // 添加CUDA设备
        const std::string &cuda_provider = "CUDAExecutionProvider";
        bool isCudaAvailable = std::find(execute_providers.begin(),
                                         execute_providers.end(),
                                         cuda_provider) != execute_providers.end();

        // 目标设备可用
        if (isCudaAvailable)
        {
            try
            {
                // 设置CUDA选项
                OrtCUDAProviderOptions cuda_options{};
                cuda_options.device_id = kGPUId;
                cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic; // 快速启动
                cuda_options.gpu_mem_limit = kMaxGPUMemLimit * 1024 * 1024 * 1024;     // 限制 4GB
                cuda_options.do_copy_in_default_stream = 1;
                session_options.AppendExecutionProvider_CUDA(cuda_options);
                LOG_INFO("[EP] CUDA registered");
            }
            catch (const std::exception &e)
            {
                LOG_WARN("[EP] CUDA unavailable: {}", e.what());
            }
        }
#endif

        // TensorRT开启会异常，优化算子会推理异常
        if (!isTensorRTAvailable) {
            // 优化路径不存在则进行模型生成
            if (!fs::exists(optimized_model_path))
            {
                // 这个过程不能发生在TensorRT推理，否则会发生异常
                session_options.SetOptimizedModelFilePath(STRING_TO_WSTRING(optimized_model_path).c_str());
            }

            // 如果优化模型路径能够被加载，直接加载优化路径
            if (fs::exists(optimized_model_path))
            {
                model_path = optimized_model_path;
            }
            else if (!fs::exists(model_path))
            {
                throw std::runtime_error("Model not found: " + model_path);
            }
        }

        try
        { // OrtSession 初始化
            LOG_INFO("Model load path: {}", model_path);
            session_ = std::make_unique<Ort::Session>(env_, MODEL_PATH(model_path).c_str(), session_options);
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("OrtSession create failed! " + std::string(e.what()));
        }

        // 获取当前推理设备信息
        ResolveActiveDevice();

        // 获取模型信息
        size_t input_names_num = session_->GetInputCount();
        LOG_TRACE("input_names_num: {}", input_names_num);
        size_t output_names_num = session_->GetOutputCount();
        LOG_TRACE("output_names_num: {}", output_names_num);

        input_names_.reserve(input_names_num);
        output_names_.reserve(output_names_num);

        Ort::AllocatorWithDefaultOptions ort_allocator;
        for (int i = 0; i < input_names_num; ++i)
        {
            input_names_.push_back(session_->GetInputNameAllocated(i, ort_allocator).get());
        }

        for (int i = 0; i < output_names_num; ++i)
        {
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
    void ResolveActiveDevice()
    {
        // 按优先级从高到低探测
        struct DeviceCandidate
        {
            const char *name;
            int device_id;
        };

        // TODO: 此处硬编码，待解决
        DeviceCandidate candidates[] = {
            {"Tensorrt", kGPUId},
            {"Cuda", kGPUId},
        };

        for (const auto &candidate : candidates)
        {
            try
            {
                Ort::MemoryInfo mem_info{candidate.name, OrtDeviceAllocator,
                                         candidate.device_id, OrtMemTypeDefault};
                // 关键：用 Session 构造 Allocator
                // 如果该设备未被 Session 接受，这里会抛异常
                allocator_ = std::make_unique<Ort::Allocator>(*session_, mem_info);
                // Ort::Allocator allocator(*session_, mem_info);
                // Ort::Allocator 提供了 GetInfo() 方法,包含了该分配器绑定的设备类型、设备ID、内存类型和分配策略。

                active_mem_info_ = std::move(mem_info);

                setActivateGPUId(candidate.device_id);
                enableGPUActivate();
                LOG_INFO("[Device] ✅ Active EP: {}, logical device {}", candidate.name, activateGPUId());
                return; // 找到最高优先级的可用设备，立即返回
            }
            catch (std::exception &e)
            {
                // 该设备不可用，继续尝试下一个
                LOG_WARN("[Device] {} not active, trying next... err: {}", candidate.name, e.what());
            }
        }

        // 所有 GPU 候选都失败
        LOG_WARN("[Device] ⚠️  All GPU EPs unavailable, fallback to CPU");
    }

    const std::vector<int64_t> &shapes() const override
    {
        return input_shapes_;
    }

    // 初始化张量，使得后续的推理全部使用该张量上进行
    void input_tensor_init()
    {
        init();
        // 如果GPU可用
        if (isGPUActivate())
        {
            // 缓存推理的数据张量
            input_tensor_ = Ort::Value::CreateTensor<float>(
                *allocator_,
                input_shapes_.data(),
                input_shapes_.size());

            pdata_ = input_tensor_.GetTensorMutableData<float>();
            input_tensors_.insert_or_assign(pdata_, std::move(input_tensor_));

#ifdef ENABLE_CUDA
            const auto& shape = shapes();
            int64_t num_elements = shape[1] * shape[2] * shape[3];
            for (int i = 0; i < pool_->capacity(); ++i) {
                float* data = pool_->Acquire();

                input_tensors_.insert_or_assign(data, 
                    Ort::Value::CreateTensor<float>(
                        active_mem_info_,
                        (float *)data,
                        num_elements,
                        shapes().data(),
                        shapes().size()
                    )
                );
    
                // input_tensors_[data] = Ort::Value::CreateTensor<float>(
                //     active_mem_info_,
                //     (float *)data,
                //     num_elements,
                //     shapes().data(),
                //     shapes().size()
                // );

                LOG_TRACE("{} tensor is valid: {}", i, input_tensors_[data].IsTensor());
    
                pool_->Release(data);
            }
#endif
        }
        else
        {
            auto &tensor_buffer = tensorBuffer();
            Ort::MemoryInfo ort_memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
            input_tensor_ = Ort::Value::CreateTensor<float>(
                ort_memory_info,
                (float *)tensor_buffer.data,
                tensor_buffer.num_elements,
                tensor_buffer.shape.data(),
                tensor_buffer.shape.size());

            pdata_ = input_tensor_.GetTensorMutableData<float>();
            LOG_TRACE("pdata addr: {}", fmt::ptr(pdata_));
            input_tensors_.insert_or_assign(pdata_, std::move(input_tensor_));
        }
    }

    // 推理无需数据，因为推理过程使用的是推理器的缓存好的空间，只需要将数据拷贝到这里即可，而这由基类完成
    ModelOutput infer() override {
        auto& tensor_buffer = tensorBuffer();
        return infer(tensor_buffer);
    }


    /// @brief 通过传入的数据进行推理
    /// @param tenbuf 预处理好的数据
    /// @return 推理结果
    ModelOutput infer(const TensorBuffer& tenbuf) override {

// #ifdef ENABLE_CUDA
//         if (isGPUActivate()) {
//             // 使用异步stream进行数据拷贝时，使用内存屏障来保证GPU数据同步
//             cudaDeviceSynchronize();
//         }
// #endif

        // 是因为热身用的随机tensor
// #ifdef ENABLE_CUDA

        if (input_tensors_.find(tenbuf.data) == input_tensors_.end()) {
            throw std::runtime_error("not found");
        }
        auto& input_tensor = input_tensors_.at(tenbuf.data);
        LOG_TRACE("tensor valid: {}", input_tensor.IsTensor());
// #else
//         auto& input_tensor = input_tensor_;
// #endif       

        // 进行推理
        auto timer = ScopedTimer("InferTimer");
        std::vector<Ort::Value> ort_outputs = session_->Run(
            Ort::RunOptions{},
            input_names_c_str_.data(),
            &input_tensor,
            input_names_c_str_.size(), // 单个节点
            output_names_c_str_.data(),
            output_names_c_str_.size());
        LOG_DEBUG("Model infer run spends: {} ms", timer.elapsed_ms());

        // 推理完成回收数据
#ifdef ENABLE_CUDA
        LOG_TRACE("Run After");
        pool_->Release(tenbuf.data);
        LOG_TRACE("TensorBuffer pool Release addr: {}", fmt::ptr(tenbuf.data));
#endif

        // ========== 3. Ort::Value → TensorBuffer ==========
        ModelOutput result;
        for (size_t i = 0; i < ort_outputs.size(); ++i)
        {
            auto &val = ort_outputs[i];
            auto type_info = val.GetTensorTypeAndShapeInfo();

            // 提取 shape
            std::vector<int64_t> shape = type_info.GetShape();

            // 零拷贝包装：不复制数据，但绑定 Ort::Value 的生命周期
            // ort_outputs 会被 move 到 lambda 捕获中，保证指针有效
            float *data_ptr = val.GetTensorMutableData<float>();

            // 创建生命周期守卫，防止 Ort::Value 提前析构
            auto guard = std::shared_ptr<Ort::Value>(
                new Ort::Value(std::move(val)),
                [](Ort::Value *p){ delete p; }); // deleter

            result.tensors[output_names_[i]] = TensorBuffer::wrap(
                data_ptr, shape,
                std::reinterpret_pointer_cast<void>(guard));

            // 此处赋值不合理
            result.tensors[output_names_[i]].letterbox_params = tenbuf.letterbox_params;
            LOG_TRACE("param: {}", tenbuf.letterbox_params[0].scale);
        }

        return result;
    }
};