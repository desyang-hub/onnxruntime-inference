#pragma once

#include <vector>
#include <future>
#include <thread>
#include <opencv2/opencv.hpp>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <yaml-cpp/yaml.h>
#include <thread>
#include <cassert>

#include "CPUBuffer.h"
#include "GPUBuffer.h"
#include "preprocess/utils.h"
#include "runner/detect/Detector.h"
#include "thread/ThreadPool.h"
#include "TaskContext.h"
#include "device/cuda_utils.h"

#include "ScopedTimer.h"
#include "queue/ConcurrentQueue.h"

/// @brief 推理流水线
class SafeQueuePipLine
{
private:
    // std::vector<std::shared_ptr<CPUBuffer>> h_inputs_;
    // std::vector<std::shared_ptr<CPUBuffer>> h_outputs_;
    // std::vector<std::shared_ptr<GPUBuffer>> d_buffers_;

    int kTimeoutMiniSeconds{2};

    using InputType     = cv::Mat;
    using OutputType    = std::vector<Detection>;

    CudaStream h2d_stream;
    CudaStream compute_stream;
    CudaStream d2h_stream;
    
    // 同步事件
    CudaEvent h2d_done;
    CudaEvent compute_done;
    CudaEvent d2h_done;

    TaskContext context_;

    bool is_close_{false};

    // std::queue<std::shared_ptr<CPUBuffer>> h_inputs_;
    // std::mutex h_input_mutex_;
    // std::condition_variable h_input_condition_;

    // std::queue<std::shared_ptr<CPUBuffer>> h_outputs_;
    // std::mutex h_output_mutex_;
    // std::condition_variable h_output_condition_;

    // std::queue<std::shared_ptr<GPUBuffer>> d_buffers_;
    // std::mutex d_buffer_mutex_;
    // std::condition_variable d_buffer_condition_;

    ConcurrentQueue<std::shared_ptr<CPUBuffer>> h_inputs_;
    ConcurrentQueue<std::shared_ptr<CPUBuffer>> h_outputs_;
    ConcurrentQueue<std::shared_ptr<GPUBuffer>> d_buffers_;
    

    // 用户维护请求队列
    std::queue<InputType> inputs_;
    std::queue<std::shared_ptr<std::promise<OutputType>>> promises_;
    std::condition_variable inputs_condition_;
    std::mutex inputs_mutex_;
    std::thread worker_;
    

    ThreadPool pre_process_{4};
    ThreadPool infer_{1};
    ThreadPool post_process_{4};


    /// @brief 预处理线程做的事情
    // void preprocess_worker(const InputType& input, std::shared_ptr<std::promise<OutputType>> pms) {
    //     // 获取可用的h_input CPUBuffer
    //     std::unique_lock<std::mutex> lock(h_input_mutex_);

    //     h_input_condition_.wait(lock, [this](){
    //         return !h_inputs_.empty() || is_close_;
    //     });

    //     // 有任务提交任务
    //     if (!h_inputs_.empty()) {
    //         std::shared_ptr<CPUBuffer> h_input = h_inputs_.front();
    //         h_inputs_.pop();
    //         lock.unlock();
            
    //         std::vector<LetterboxParams> params;
    //         try {
    //             params.push_back(preprocess(input, h_input->data()));
    //         } catch (...) {
    //             pms->set_exception(std::current_exception());
    //             {
    //                 std::lock_guard<std::mutex> lock(h_input_mutex_);
    //                 h_inputs_.push(h_input);
    //             }
    //             h_input_condition_.notify_one();
    //         }
    //         // 预处理，提交推理
    //         infer_.enqueue([this, params, pms, h_input](){
    //             // 获取可用的GPUBuffer
    //             std::unique_lock<std::mutex> lock(d_buffer_mutex_);
    //             d_buffer_condition_.wait(lock, [this](){
    //                 return !d_buffers_.empty() || is_close_;
    //             });

    //             // 处理kernel任务
    //             if (!d_buffers_.empty()) {
    //                 std::shared_ptr<GPUBuffer> d_buf = d_buffers_.front();
    //                 d_buffers_.pop();
    //                 lock.unlock();

    //                 // 准备output CPUBuffer
    //                 std::unique_lock<std::mutex> lock(h_output_mutex_);
    //                 h_output_condition_.wait(lock, [this](){
    //                     return !h_outputs_.empty() || is_close_;
    //                 });

    //                 bool err = true;
    //                 if (!h_outputs_.empty()) {
    //                     auto h_output = std::move(h_outputs_.front());
    //                     h_outputs_.pop();
    //                     lock.unlock();

    //                     try
    //                     {
    //                         kernel(h_input, d_buf, h_output);
    //                         // 结束后开始后处理任务
    //                         post_process_.enqueue([this, h_output, pms, params, d_buf](){
    //                             cudaEventSynchronize(d2h_done.get());
    //                             // d2h 完成释放GPU Buffer
    //                             {
    //                                 std::lock_guard<std::mutex> lock(d_buffer_mutex_);
    //                                 d_buffers_.push(d_buf);
    //                             }
    //                             d_buffer_condition_.notify_one();

    //                             std::vector<Detection> res;

    //                             bool is_err = false;
    //                             try {
    //                                 res = postprocess(h_output, params)[0];
    //                             } catch (...) {
    //                                 is_err = true;
    //                                 pms->set_exception(std::current_exception());
    //                             }

    //                             {
    //                                 std::lock_guard<std::mutex> lock(h_output_mutex_);
    //                                 h_outputs_.push(std::move(h_output));
    //                             }
    //                             h_output_condition_.notify_one();
    //                             if (is_err) return;

    //                             pms->set_value(res);
    //                         });
    //                         err = false;
    //                     }
    //                     catch(...)
    //                     {
    //                         pms->set_exception(std::current_exception());
    //                         {
    //                             std::lock_guard<std::mutex> lock(h_output_mutex_);
    //                             h_outputs_.push(h_output);
    //                         }
    //                         h_output_condition_.notify_one();
    //                     }
    //                 } 
                    
    //                 if (err) {
    //                     {
    //                         std::lock_guard<std::mutex> lock(d_buffer_mutex_);
    //                         d_buffers_.push(d_buf);
    //                     }
    //                     d_buffer_condition_.notify_one();
    //                 }
    //             }
    //         });
    //     }
    
    // }


    void bufferInit() {
        for (int i = 0; i < context_.buffer_size; ++i) {
            h_inputs_.push(std::make_shared<CPUBuffer>(context_.num_input_bytes_size));
            h_outputs_.push(std::make_shared<CPUBuffer>(context_.num_output_bytes_size));

            auto d_buffer = std::make_shared<GPUBuffer>(context_.num_input_elements, context_.num_output_elements);

            d_buffer->input_tensor = Ort::Value::CreateTensor<float>(
                context_.active_mem_info,
                d_buffer->g_input.get(),
                context_.num_input_elements,
                context_.input_shape.data(),
                context_.input_shape.size()
            );

            d_buffer->output_tensor = Ort::Value::CreateTensor<float>(
                context_.active_mem_info,
                d_buffer->g_output.get(),
                context_.num_output_elements,
                context_.output_shape.data(),
                context_.output_shape.size()
            );
            d_buffers_.push(std::move(d_buffer));
            
        }
    }


    void submit_worker() {
        while (true) {
            std::unique_lock<std::mutex> lock(inputs_mutex_);
            bool success = inputs_condition_.wait_for(
                lock, 
                std::chrono::milliseconds(kTimeoutMiniSeconds),
                [this](){
                    return inputs_.size() >= context_.batch_size || is_close_;
                }
            );

            if ((!success && !inputs_.empty()) || inputs_.size() >= context_.batch_size) {
                do
                {
                    int patch = std::min(context_.batch_size, inputs_.size());
                    
                    std::shared_ptr<CPUBuffer> h_input;
                    h_inputs_.pop(h_input);

                    std::vector<std::future<LetterboxParams>> futs;
                    std::vector<std::shared_ptr<std::promise<OutputType>>> pmses;
                    futs.reserve(patch);
                    pmses.reserve(patch);
                    // 提交预处理任务
                    for (int i = 0; i < patch; ++i) {
                        InputType inp = std::move(inputs_.front());
                        inputs_.pop();
                        pmses.push_back(std::move(promises_.front()));
                        promises_.pop();

                        futs.push_back(
                            pre_process_.enqueue([this, inp, h_input, i]{
                                return preprocess(inp, h_input->data() + i * context_.input_plane_size);
                            })
                        );
                    }

                    // 等待预处理完成开始推理
                    std::vector<LetterboxParams> params;
                    params.reserve(patch);
                    for (auto& fut : futs) params.push_back(fut.get());

                    // 获取GPUBuffer
                    std::shared_ptr<GPUBuffer> d_buf;
                    d_buffers_.pop(d_buf);

                    // 获取output CPU BUffer
                    std::shared_ptr<CPUBuffer> h_output;
                    h_outputs_.pop(h_output);
                    
                    infer_.enqueue([this, pmses, h_input, d_buf, h_output, params=std::move(params)]{
                        kernel(h_input, d_buf, h_output);

                        // 后处理
                        post_process_.enqueue([this, pmses, d_buf, h_output, params=std::move(params)]{
                            cudaEventSynchronize(d2h_done.get());
                            // 还资源
                            d_buffers_.push(std::move(d_buf));

                            // 启动后处理
                            std::vector<OutputType> opts = postprocess(h_output, params);

                            // 还资源
                            h_outputs_.push(std::move(h_output));

                            assert(opts.size() == pmses.size());

                            for (int i = 0; i < opts.size(); ++i) {
                                pmses[i]->set_value(std::move(opts[i]));
                            }
                        });

                    });

                } while (!inputs_.empty());
            }
            else if (is_close_) {
                break;
            }
        }
    }

    public:
    SafeQueuePipLine(const YAML::Node& config, 
        const Ort::Env& env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "OrtDefaultLogid")) : 
        context_(config, env, compute_stream.get()) {
        bufferInit();
        context_.warm_up(*d_buffers_.front());

        worker_ = std::thread(std::bind(&SafeQueuePipLine::submit_worker, this));
    }

    ~SafeQueuePipLine() {
        {
            std::lock_guard<std::mutex> lock(inputs_mutex_);
            is_close_ = true;
        }
        inputs_condition_.notify_all();

        if (worker_.joinable()) worker_.join();
    }

    /// @brief 预处理过程
    /// @param img 数据源
    /// @param dst h_input
    /// @return 后处理需要用的参数
    LetterboxParams preprocess(const cv::Mat& img, float* dst) {

        ScopedTimer st("pre");
        
        // 所需要的参数应当通过上下文传入
        cv::Mat resized;
        // b, c, h, w
        LetterboxParams params = 
            letterbox_resize(img, resized, context_.img_size, context_.auto_aspect_ratio);
        
        convert_and_normalize(resized, dst, context_.img_size, context_.norm_scale, context_.bgr2rgb);

        LOG_INFO("pre time: {}", st.elapsed_ms());
    
        return params;
    }

    // GPU kernel 过程
    void kernel(
        std::shared_ptr<CPUBuffer> h_input, 
        std::shared_ptr<GPUBuffer> d_buffer, 
        std::shared_ptr<CPUBuffer> h_output ) { 

        ScopedTimer st("kernel");
        // h2d 并设置状态
        cudaMemcpyAsync(
            d_buffer->g_input.get(), 
            h_input->data(),                        // ✅ 修复1
            context_.num_input_bytes_size, 
            cudaMemcpyHostToDevice, 
            h2d_stream.get()
        );
        cudaEventRecord(h2d_done.get(), h2d_stream.get()); // 纪录事件完成

        // compute 并设置状态
        cudaStreamWaitEvent(compute_stream.get(), h2d_done.get());
        context_.session->Run(
            Ort::RunOptions{}, 
            context_.input_names_c_str.data(), &d_buffer->input_tensor, context_.input_names_c_str.size(),
            context_.output_names_c_str.data(), &d_buffer->output_tensor, context_.output_names_c_str.size()
        );
        cudaEventRecord(compute_done.get(), compute_stream.get());

        // d2h 并设置状态
        cudaStreamWaitEvent(d2h_stream.get(), compute_done.get());

        // compute 完成，回收h_input
        h_inputs_.push(std::move(h_input));

        cudaMemcpyAsync(
            h_output->data(), 
            d_buffer->g_output.get(), 
            context_.num_output_bytes_size, 
            cudaMemcpyDeviceToHost, 
            d2h_stream.get()
        );
        cudaEventRecord(d2h_done.get(), d2h_stream.get());

        LOG_INFO("kernel time: {}", st.elapsed_ms());
    }

    /// @brief 后处理过程
    /// @param h_output 推理结果
    /// @return 输出后处理结果
    std::vector<std::vector<Detection>> postprocess(
        std::shared_ptr<CPUBuffer> h_output, 
        const std::vector<LetterboxParams>& params) {

        ScopedTimer st("post");

        int patch = params.size();
        // YOLOv8 输出形状: [1, numAttributes(4+num_classes), numPredictions]
        std::vector<std::vector<Detection>> results(patch);
    
    // #pragma omp parallel for schedule(dynamic)
        for (int b = 0; b < patch; ++b) { // 只需要处理实际的批量大小即可
            const float* pdata = h_output->data() + b * context_.output_plane_size;
            std::vector<Detection> result;
        
            // ========== 2. 解码候选框 (替代原始的 transpose + row遍历) ==========
            // 💡 优化：直接在原始 [84, 8400] 布局上按列访问，避免 cv::transpose 的内存拷贝开销
            std::vector<cv::Rect2i> boxes;
            std::vector<float> confidences;
            std::vector<int> class_ids;
            boxes.reserve(context_.max_detections);
            confidences.reserve(context_.max_detections);
            class_ids.reserve(context_.max_detections);
            
            for (int i = 0; i < context_.num_predictions; ++i) {
                // 在 [84, 8400] 布局中，第 i 个预测框的数据起始位置
                // cx=pdata[i], cy=pdata[N+i], w=pdata[2N+i], h=pdata[3N+i]
                const float* box_ptr = pdata + i;
                const float* cls_ptr = pdata + 4 * context_.num_predictions + i;
                
                // Argmax 寻找最佳类别 (替代 minMaxLoc)
                float best_score = 0.f;
                int best_cls = 0;
                for (int c = 0; c < context_.num_classes; ++c) {
                    size_t id = c * context_.num_predictions;
                    float s = cls_ptr[c * context_.num_predictions];
                    
                    // LOG_INFO("IDX: {}", c * context_.num_predictions);
                    if (s > best_score) {
                        best_score = s;
                        best_cls = c;
                    }
                }
                
                if (best_score < context_.conf_threshold) continue;
                
                // 提取 cx, cy, w, h
                float cx = box_ptr[0];
                float cy = box_ptr[context_.num_predictions];
                float ow = box_ptr[2 * context_.num_predictions];
                float oh = box_ptr[3 * context_.num_predictions];
                
                // ⭐ 坐标还原到原图 (替代原始的 x_factor 乘法)
                float x1 = std::clamp((cx - 0.5f * ow - params[b].pad_left) / params[b].scale, 0.f, (float)params[b].orig_w);
                float y1 = std::clamp((cy - 0.5f * oh - params[b].pad_top)  / params[b].scale, 0.f, (float)params[b].orig_h);
                float x2 = std::clamp((cx + 0.5f * ow - params[b].pad_left) / params[b].scale, 0.f, (float)params[b].orig_w);
                float y2 = std::clamp((cy + 0.5f * oh - params[b].pad_top)  / params[b].scale, 0.f, (float)params[b].orig_h);
                
                // 过滤无效框
                if (x2 <= x1 || y2 <= y1) {
                    continue;
                } 
                
                // 转为 cv::Rect 用于 NMS (OpenCV NMSBoxes 要求 Rect)
                boxes.emplace_back(
                    static_cast<int>(std::round(x1)), static_cast<int>(std::round(y1)),
                    static_cast<int>(std::round(x2 - x1)), static_cast<int>(std::round(y2 - y1))
                );
                confidences.push_back(best_score);
                class_ids.push_back(best_cls);
            }
            
            // ========== 3. NMS ==========
            std::vector<int> indexes;
            if (!boxes.empty()) {
                cv::dnn::NMSBoxes(boxes, confidences, context_.conf_threshold, context_.nms_threshold, indexes);
            }
            
            // ========== 4. 组装最终结果 ==========
            result.reserve(indexes.size());
            for (int idx : indexes) {
                result.push_back({
                    boxes[idx],
                    confidences[idx],
                    class_ids[idx]
                });
            }
    
            // 将并行结果移动到结果中
            results[b] = std::move(result);
        }

        LOG_INFO("post time: {}", st.elapsed_ms());
        
        return results;
    }


    // std::future<OutputType> submit(const InputType& input) {

    //     auto pms = std::make_shared<std::promise<OutputType>>();
    //     auto fut = pms->get_future();

    //     pre_process_.enqueue(std::bind(&SafeQueuePipLine::preprocess_worker, this, input, pms));

    //     return fut;
    // }

    std::future<OutputType> submit(const InputType& input) {

        auto pms = std::make_shared<std::promise<OutputType>>();
        auto fut = pms->get_future();

        // 数据入队
        {
            std::lock_guard<std::mutex> lock(inputs_mutex_);
            inputs_.push(input);
            promises_.push(pms);
        }
        inputs_condition_.notify_one();

        return fut;
    }

    // OutputType detect(const InputType& input) {
    //     std::shared_ptr<CPUBuffer> h_input;
    //     LetterboxParams param = preprocess(input, h_input->data());

    //     std::shared_ptr<GPUBuffer> d_bufer = 
    //     kernel()
    // }
};