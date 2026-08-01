#pragma once

#include <onnxruntime_cxx_api.h>
#include <cuda_runtime.h>
#include <memory>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <numeric>
#include <atomic>

#include "device/cuda_utils.h"
#include "arch/arch.h"
#include "TensorBuffer.h"
#include "logger/logger.h"
#include "GPUBuffer.h"
#include "onnx/utils.h"
#include "logger/logger.h"


struct TaskContext
{
    // session
    std::unique_ptr<Ort::Session> session;

    std::unique_ptr<Ort::Allocator> allocator;
    Ort::MemoryInfo active_mem_info{nullptr};
    Ort::MemoryInfo cpu_mem_info{nullptr};

    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<const char*> input_names_c_str;
    std::vector<const char*> output_names_c_str;
    cudaStream_t user_compute_stream_{nullptr}; // 用户自己选择是否需要


    // =========================== config ===========================
    cv::Size img_size;
    bool auto_aspect_ratio{true};
    float norm_scale{1.0f/255.0f};
    int pad_value{127};
    bool bgr2rgb{true};

    size_t num_input_elements{0};
    size_t num_input_bytes_size{0}; // * sizeof(flaot)
    size_t num_output_elements{0};
    size_t num_output_bytes_size{0};// * sizeof(flaot)

    size_t num_predictions{0};
    size_t num_attributes{0};
    size_t num_classes{0};
    size_t batch_size;
    size_t output_plane_size; // total_elements / batch_size
    size_t input_plane_size;

    size_t max_detections{300}; // max deteciton num
    float conf_threshold;
    float nms_threshold;

    bool enable_profiling;
    std::string model_path;
    std::string optimized_model_path;
    std::vector<std::string> execute_providers;
    int gpu_id{-1};
    std::string FP16_enable{"0"};
    int warm_up_num{3};

    int buffer_size{1};
    int stride{32};
    
    TaskContext(const YAML::Node& config, 
        const Ort::Env& env, cudaStream_t user_compute_stream=nullptr) : 
        user_compute_stream_(user_compute_stream) {
        init(config, env);
        cpu_mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    }

    /// @brief OrtSession 初始化
    /// @param config 配置项目
    /// @param env Ort::Env
    void init(const YAML::Node& config, const Ort::Env& env) {
        // Ort::SessionOptions 配置
        Ort::SessionOptions sessionoptions;
        
        // 从配置文件中获取sessionOptions配置
        model_path = 
            config["model"]["path"].as<std::string>();
        std::vector<std::string> execution_providers =
            config["model"]["session_options"]["execution_providers"]
                .as<std::vector<std::string>>(std::vector<std::string>{"CPUExecutionProvider"});
        std::string graph_optimization_level =
            config["model"]["session_options"]["graph_optimization_level"]
                .as<std::string>("ORT_ENABLE_BASE");
        int intra_op_num_threads =
            config["model"]["session_options"]["intra_op_num_threads"]
                .as<int>(1);
        int inter_op_num_threads =
            config["model"]["session_options"]["inter_op_num_threads"]
                .as<int>(1);
        std::string execution_mode =
            config["model"]["session_options"]["execution_mode"]
                .as<std::string>("ORT_SEQUENTIAL");
        std::string log_severity_level =
            config["model"]["session_options"]["log_severity_level"]
                .as<std::string>("WARNING");
        enable_profiling =
            config["model"]["session_options"]["enable_profiling"].as<bool>(false);
        optimized_model_path =
            config["model"]["session_options"]["optimized_model_path"].as<std::string>("");
        execute_providers =
            config["model"]["session_options"]["execution_providers"].as<std::vector<std::string>>();
        FP16_enable = 
            config["model"]["session_options"]["trt_fp16_enable"].as<std::string>("0");


        std::vector<int64_t> img_shape =
            config["model"]["shape"].as<std::vector<int64_t>>(std::vector<int64_t>{640, 640});
        batch_size = 
            config["model"]["batch"].as<size_t>(1);
        gpu_id = 
            config["model"]["gpu"].as<int>(0);
        buffer_size =
            config["model"]["buffer_size"].as<int>(1);
            
        warm_up_num = 
            config["model"]["warm_up"].as<size_t>(0);


        // postprocess configuration
        conf_threshold = 
            config["postprocess"]["conf_threshold"].as<float>(0.25);
        nms_threshold = 
            config["postprocess"]["nms_threshold"].as<float>(0.45);
        max_detections = 
            config["postprocess"]["max_detections"].as<size_t>(300);

        // attribute set
        img_size.width = img_shape[0];
        img_size.height = img_shape[1];

        // 设置sessionOptions
        sessionoptions.SetGraphOptimizationLevel(ParseGraphOptimizationLevel(graph_optimization_level));
        sessionoptions.SetIntraOpNumThreads(intra_op_num_threads);
        sessionoptions.SetInterOpNumThreads(inter_op_num_threads);
        sessionoptions.SetExecutionMode(ParseExecutionMode(execution_mode));
        sessionoptions.SetLogSeverityLevel(ParseLogSeverityLevel(log_severity_level));
        sessionoptions.EnableMemPattern(); // 必须开启，Graph 依赖内存复用
        // sessionoptions.AddConfigEntry("session.dynamic_shape_optimization", "0");

        // EP Device register
        epDeviceRegister(sessionoptions);

        // OrtSession 初始化
        try {
            LOG_INFO("Model load path: {}", model_path);
            session = std::make_unique<Ort::Session>(env, MODEL_PATH(model_path).c_str(), sessionoptions);
        } catch (const std::exception &e) {
            throw std::runtime_error(MESSAGE_WITH_LOC("OrtSession create failed! " + std::string(e.what())));
        }

        resolveActiveDevice(gpu_id);

        modelInit();

        // num_predictions = 
    }

    void epDeviceRegister(Ort::SessionOptions& sessionoptions) {
        if (enable_profiling) {
            sessionoptions.EnableProfiling(STRING_TO_WSTRING("profiling").c_str());
        }

        namespace fs = std::filesystem;
        if (optimized_model_path.empty()) {
            // 配置路径
            fs::path p(model_path);
            // 提取文件名（不含扩展名）+ 新前缀 + 原扩展名
            fs::path new_path = p.parent_path() / ("optimized_" + p.stem().string() + p.extension().string());
            optimized_model_path = new_path.string();
        }

        // 获取环境中可用的推理设备
        // auto execute_providers = Ort::GetAvailableProviders();
        bool isTensorRTAvailable = false;

#ifdef ENABLE_CUDA
        static constexpr unsigned long long kMaxGPUMemLimit = 8; // 限制4GB显存空间

        const std::string &tensorrt_provider = "TensorrtExecutionProvider";
        isTensorRTAvailable = std::find(execute_providers.begin(),
                                        execute_providers.end(),
                                        tensorrt_provider) != execute_providers.end();

        // 使用V2 api开启TensorRt
        if (isTensorRTAvailable) {
            try {
                // 1. 获取全局 C API 指针（所有版本都有这个函数）
                const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
                if (!ort_api) {
                    throw std::runtime_error(MESSAGE_WITH_LOC("Failed to get ONNX Runtime C API"));
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
                std::string device_id_str  = std::to_string(gpu_id);
        
                const char* keys[] = {
                    "device_id",
                    "trt_max_workspace_size",
                    "trt_fp16_enable",
                    "trt_engine_cache_enable",
                    "trt_engine_cache_path"
                };

                

                const char* values[] = {
                    device_id_str.c_str(),
                    workspace_size.c_str(),
                    FP16_enable.c_str(),   // FP16 ON["1"], OFF["0"]
                    "1",   // Cache ON
                    "./trt_cache"
                };
        
                status = ort_api->UpdateTensorRTProviderOptions(trt_opts, keys, values, sizeof(keys) / sizeof(keys[0]));
                if (status != nullptr) {
                    std::string err = ort_api->GetErrorMessage(status);
                    ort_api->ReleaseStatus(status);
                    ort_api->ReleaseTensorRTProviderOptions(trt_opts);
                    throw std::runtime_error(MESSAGE_WITH_LOC("UpdateTensorRTProviderOptions failed: " + err));
                }
        
                // 4. 挂载到 C++ SessionOptions（C/C++ 混合调用关键步骤）
                sessionoptions.AppendExecutionProvider_TensorRT_V2(*trt_opts);
        
                // 5. 释放 V2 Options
                ort_api->ReleaseTensorRTProviderOptions(trt_opts);
                
                if (FP16_enable == "1")
                    LOG_INFO("[EP] TensorRT registered (FP16 enabled via C API)");
                else
                    LOG_INFO("[EP] TensorRT registered (FP32 enabled via C API)");
            } catch (const std::exception &e) {
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
        if (isCudaAvailable && !isTensorRTAvailable) {
            try {
                // 设置CUDA选项
                OrtCUDAProviderOptions cuda_options{};
                cuda_options.device_id = gpu_id;
                cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic; // 快速启动
                cuda_options.gpu_mem_limit = kMaxGPUMemLimit * 1024 * 1024 * 1024;     // 限制 4GB
                cuda_options.do_copy_in_default_stream = 1;

                // 启用使用用户自定义流
                if (user_compute_stream_) {
                    cuda_options.has_user_compute_stream = 1;
                    cuda_options.user_compute_stream = user_compute_stream_; // 使用自定义stream
                }
                
                sessionoptions.AppendExecutionProvider_CUDA(cuda_options);
                LOG_INFO("[EP] CUDA registered");
            } catch (const std::exception &e) {
                LOG_WARN("[EP] CUDA unavailable: {}", e.what());
                isCudaAvailable = false;
            }
        }
#endif

        // TensorRT开启会异常，优化算子会推理异常
        if (!isTensorRTAvailable) {
            // 优化路径不存在则进行模型生成
            if (!fs::exists(optimized_model_path))
            {
                // 这个过程不能发生在TensorRT推理，否则会发生异常
                sessionoptions.SetOptimizedModelFilePath(STRING_TO_WSTRING(optimized_model_path).c_str());
            }

            // 如果优化模型路径能够被加载，直接加载优化路径
            if (fs::exists(optimized_model_path))
            {
                model_path = optimized_model_path;
            }
            else if (!fs::exists(model_path))
            {
                throw std::runtime_error(MESSAGE_WITH_LOC("Model not found: " + model_path));
            }
        }
    }


    void resolveActiveDevice(int gpu_id) {
        // 按优先级从高到低探测
        struct DeviceCandidate
        {
            const char *name;
            int device_id;
        };

        // TODO: 此处硬编码，待解决
        DeviceCandidate candidates[] = {
            {"Tensorrt", gpu_id},
            {"Cuda", gpu_id},
        };

        for (const auto &candidate : candidates)
        {
            try
            {
                active_mem_info = Ort::MemoryInfo{
                    candidate.name, 
                    OrtDeviceAllocator, 
                    candidate.device_id, 
                    OrtMemTypeDefault
                };

                // 关键：用 Session 构造 Allocator
                // 如果该设备未被 Session 接受，这里会抛异常
                allocator = std::make_unique<Ort::Allocator>(*session, active_mem_info);
                LOG_INFO("[Device] ✅ Active EP: {}, logical device {}", candidate.name, candidate.device_id);

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
    

    void modelInit() {
        // 获取模型信息
        size_t input_names_num = session->GetInputCount();
        LOG_TRACE("input_names_num: {}", input_names_num);
        size_t output_names_num = session->GetOutputCount();
        LOG_TRACE("output_names_num: {}", output_names_num);

        input_names.reserve(input_names_num);
        output_names.reserve(output_names_num);

        Ort::AllocatorWithDefaultOptions ort_allocator;
        for (int i = 0; i < input_names_num; ++i)
        {
            input_names.push_back(session->GetInputNameAllocated(i, ort_allocator).get());
        }

        for (int i = 0; i < output_names_num; ++i)
        {
            output_names.push_back(session->GetOutputNameAllocated(i, ort_allocator).get());
        }

        input_names_c_str.push_back(input_names[0].c_str());
        output_names_c_str.push_back(output_names[0].c_str());

        std::vector<int64_t> raw_input_shapes =
            session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

        // batch 设置
        input_shape = parse_input_meta(raw_input_shapes, {img_size.height, img_size.width});
        input_shape[0] = batch_size; // batch
        num_input_elements = std::accumulate(input_shape.begin(), input_shape.end(), size_t{1}, std::multiplies<size_t>());
        input_plane_size = num_input_elements / batch_size;
        num_input_bytes_size = num_input_elements * sizeof(float);
        output_shape = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

        
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::Value dummy_tensort = Ort::Value::CreateTensor<float>(
            allocator,
            input_shape.data(),
            input_shape.size()
        );;

        std::vector<Ort::Value> dummy_opt = session->Run(
            Ort::RunOptions{}, 
            input_names_c_str.data(),
            &dummy_tensort,
            1,
            output_names_c_str.data(),
            output_names_c_str.size()
        );
        output_shape = dummy_opt[0].GetTensorTypeAndShapeInfo().GetShape();
        num_output_elements = std::accumulate(output_shape.begin(), output_shape.end(), size_t{1}, std::multiplies<size_t>());
        num_output_bytes_size = num_output_elements * sizeof(float);

        num_predictions = output_shape[2];
        num_attributes = output_shape[1];
        num_classes = num_attributes - 4;
        output_plane_size = num_output_elements / batch_size;
    }

    /// @brief 进行模型热身，并通过输出来设置output_shape
    void warm_up(GPUBuffer& gpu_buffer) {
        for (int i = 0; i < warm_up_num; ++i) {
            session->Run(
                Ort::RunOptions{}, 
                input_names_c_str.data(), &gpu_buffer.input_tensor, input_names_c_str.size(),
                output_names_c_str.data(), &gpu_buffer.output_tensor, output_names_c_str.size()
            );        
        }
        LOG_INFO("Model warm up success.");
    }
};

