#pragma once

#include "IBench.h"
#include "kernels/yolo/preprocess.cuh"
#include "kernels/yolo/postprocess.cuh"
#include "thread/ThreadPool.h"
#include "exceptions/utils.h"
#include "device/cuda_utils.h"
#include "queue/ConcurrentQueue.h"

#include <stdexcept>
#include <vector>
#include <future>
#include <condition_variable>
#include <chrono>
#include <thread>

class BenchKernelBatchPipline;

// === Per-Request 上下文 ===
struct InferRequestContext {

    InferRequestContext(const InferRequestContext&) = delete;
    InferRequestContext& operator=(const InferRequestContext&) = delete;

    InferRequestContext(InferRequestContext&&) = default;
    InferRequestContext& operator=(InferRequestContext&&) = default;

    static const int kAttributeSize{6}; // x, y, w, h, class_id, score
    static const int kMaxImage{1024ULL * 1024 * 3}; // h * w * channels

    // GPU 资源
    CudaStream h2d_stream;
    CudaEvent h2d_done;
    CudaStream d2h_stream;
    CudaEvent d2h_done;

    // ⭐ 反向引用 Pipeline（用于提交后处理、归还 buffer 等）
    BenchKernelBatchPipline* pipeline{nullptr};
    
    // Buffer slot（从池中分配）
    GPUBuffer d_buffer;
    CudaMallocGuard<float> d_output;
    CudaMallocGuard<int> d_counts;
    CudaMallocGuard<uint8_t> d_rgb_srcs;
    CudaMallocGuard<size_t> d_img_offsets;
    CudaMallocGuard<ImageMeta> d_metas;

    CudaMallocHostGuard<float> h_output;
    CudaMallocHostGuard<int> h_counts;
    CudaMallocHostGuard<uint8_t> h_rgb_srcs;
    CudaMallocHostGuard<size_t> h_img_offsets;
    CudaMallocHostGuard<ImageMeta> h_metas;
    
    int output_plane_size;
    
    // 业务元数据
    std::vector<LetterboxParams> letterbox_params;
    std::vector<std::shared_ptr<std::promise<Detector::OutputType>>> promises_;

    int ctx_id;
    


    /// @brief 初始化InferRequestContext
    /// @param context TaskContext
    InferRequestContext(BenchKernelBatchPipline* ben, TaskContext& context) :
        pipeline(ben),
        d_buffer(context),
        d_output(context.batch_size * context.max_detections * kAttributeSize), // d_output 是最后阶段输出
        d_counts(context.batch_size),
        d_rgb_srcs(context.batch_size * kMaxImage * sizeof(uint8_t)),
        d_img_offsets(context.batch_size),
        d_metas(context.batch_size),
        h_output(context.batch_size * context.max_detections * kAttributeSize), // d_output 是最后阶段输出
        h_counts(context.batch_size),
        h_rgb_srcs(context.batch_size * kMaxImage * sizeof(uint8_t)),
        h_img_offsets(context.batch_size),
        h_metas(context.batch_size) {}
};

// kernel下
class BenchKernelBatchPipline : IBench
{
    static constexpr int kAttributeSize{6}; // x, y, w, h, class_id, score
    static constexpr int kMaxImage{1024ULL * 1024 * 3}; // h * w * channels
    static constexpr int kTimeoutMiliseconds{2}; // 设置2ms的超时时间

    // 通过自定义流进行并发控制
    // CudaStream h2d_stream_;
    // CudaStream d2h_stream_;
    CudaStream compute_stream_;
    CudaEvent compute_done_;
    // CudaEvent h2d_done_;
    // CudaEvent d2h_done_;

    Ort::Env env_;
    TaskContext context_;

    // 流水线需要的三个任务线程
    ThreadPool preprocess_pool_{4};
    ThreadPool infer_pool_{1};
    ThreadPool postpreprocess_pool_{4};



    // int buffer_size_;

    // ConcurrentQueue<GPUBuffer> d_buffers_;
    // ConcurrentQueue<CudaMallocGuard<float>> d_outputs_;
    // ConcurrentQueue<CudaMallocGuard<int>> d_counts_;
    // ConcurrentQueue<CudaMallocGuard<uint8_t>> gpu_rgb_srcs_;
    // ConcurrentQueue<CudaMallocGuard<size_t>> gpu_img_offsets_;
    // ConcurrentQueue<CudaMallocGuard<ImageMeta>> gpu_imgMetasArry_;
    

    // ConcurrentQueue<std::unique_ptr<int[]>> h_counts_;
    // ConcurrentQueue<std::unique_ptr<size_t[]>> cpu_img_offsets_;
    // ConcurrentQueue<std::vector<ImageMeta>> imgMetasArry_;
    // ConcurrentQueue<std::vector<LetterboxParams>> letterbox_params_arry_;


    std::queue<cv::Mat> img_que_;
    std::queue<ImageMeta> img_meta_que_;
    std::queue<std::shared_ptr<std::promise<Detector::OutputType>>> promises_;
    std::mutex mutex_;
    std::condition_variable condition_;
    bool is_close_{false};
    std::thread worker_th_;

    // 缓冲区
    // ConcurrentQueue<std::shared_ptr<InferRequestContext>> ctx_buffers_;
    std::vector<InferRequestContext> ctx_buffers_;
    ConcurrentQueue<int> ctx_id_que_;


    void init() {
        int kBufferSize = 2;
        ctx_buffers_.reserve(kBufferSize);
        for (int i = 0; i < kBufferSize; ++i) {
            ctx_buffers_.emplace_back(this, context_);
            ctx_id_que_.push(i);
        } 

        context_.warm_up(ctx_buffers_[0].d_buffer);

        // 启动后台任务线程
        worker_th_ = std::thread(&BenchKernelBatchPipline::worker, this);
    }


    /// @brief 批量推理入口，外部已上锁
    void batch_infer_work() {
        int patch = std::min(context_.batch_size, img_que_.size());

        int ctx_id;
        ctx_id_que_.pop(ctx_id);
        InferRequestContext* ctx = &ctx_buffers_[ctx_id];
        ctx->ctx_id = ctx_id;

        // 进行推理参数构建
        std::vector<cv::Mat> imgs;
        imgs.reserve(patch);
        ctx->promises_.clear();
        ctx->letterbox_params.clear();
        for (int i = 0; i < patch; ++i) {
            imgs.push_back(img_que_.front());
            img_que_.pop();


            ImageMeta meta = std::move(img_meta_que_.front());
            img_meta_que_.pop();
            ctx->h_metas[i] = meta;
            ctx->letterbox_params.push_back({
                meta.scale,
                meta.pad_left,
                meta.pad_top,
                meta.src_w,
                meta.src_h
            });            

            ctx->promises_.push_back(std::move(promises_.front()));
            promises_.pop();
        }

        // 提交推理任务
        infer_pool_.enqueue([this, ctx, imgs](){
            try {
                gpu_process(imgs, *ctx);
            } catch(...)
            {
                LOG_ERROR("gpu process function call error");
                for (auto& pms : ctx->promises_) {
                    pms->set_exception(std::current_exception());
                }
            }
        });
    }

    /// @brief 后台线程，用于监控提交的任务队列
    void worker() {
        while (true) {
            {
                std::unique_lock<std::mutex> lock(mutex_);

                bool success = condition_.wait_for(
                    lock, 
                    std::chrono::milliseconds(kTimeoutMiliseconds), 
                    [this](){
                        return img_que_.size() >= context_.batch_size || is_close_;
                    }
                );

                // 如果超时有未处理图像，进行处理或者batch已经凑够直接处理
                if ((!success && !img_que_.empty()) || img_que_.size() >= context_.batch_size) {
                    do {
                        // 处理任务的逻辑
                        batch_infer_work();
                    } while (!img_que_.empty()); // 处理直到队列为空
                }

                else if (is_close_) {
                    break; // 退出任务
                }
            }

        }
    }

public:
    /// @brief 通过配置文件进行初始化构造
    /// @param config 
    BenchKernelBatchPipline(const YAML::Node& config) : 
        env_(ORT_LOGGING_LEVEL_ERROR, "BenchKernelBatchPipline"),
        context_(config, env_, compute_stream_.get()) {

        init();
    }

    /// @brief 通过TaskContext进行初始化构造
    /// @param context 
    // BenchKernelBatchPipline(TaskContext&& context) : context_(std::move(context)) {
    //     init();
    // }

    int getBatchSize() const {
        return context_.batch_size;
    }

    ~BenchKernelBatchPipline() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            is_close_ = true;
        }
        condition_.notify_all();

        if (worker_th_.joinable()) worker_th_.join(); // join thread
    }

    /// @brief 单图预处理过程
    /// @param img 
    /// @return 
    ImageMeta preprocess_v1(const cv::Mat& img) {
        // CPU上计算参数，并将数据移动到GPU上
        float scale_h = static_cast<float>(context_.img_size.height) / img.rows;
        float scale_w = static_cast<float>(context_.img_size.width) / img.cols;

        float scale = std::min(scale_h, scale_w);
        int new_h = static_cast<int>(std::round(scale * img.rows));
        int new_w = static_cast<int>(std::round(scale * img.cols));

        new_h -= (new_h % context_.stride);
        new_w -= (new_w % context_.stride);

        scale = std::min(static_cast<float>(new_w) / img.cols,
                 static_cast<float>(new_h) / img.rows);

        int pad_left = (context_.img_size.width - new_w) / 2;
        int pad_top = (context_.img_size.height - new_h) / 2;

        return ImageMeta{img.rows, img.cols, img.step, pad_left + new_w, pad_top + new_h, scale, pad_left, pad_top};

        // return LetterboxParams{scale, pad_left, pad_top, img.cols, img.rows};
    }


    /// @brief 这个函数是AsyncRun的回调函数，这个函数发生在推理结束之后
    /// @param user_data 用户数据 `InferRequestContext`
    /// @param outputs 
    /// @param num_outputs 
    /// @param status 
    static void callback_func(void* user_data, OrtValue** outputs, size_t num_outputs, OrtStatusPtr status) {
        LOG_TRACE("callback_func call");

        InferRequestContext& ctx = *(reinterpret_cast<InferRequestContext*>(user_data));

        BenchKernelBatchPipline& self = *(reinterpret_cast<BenchKernelBatchPipline*>(ctx.pipeline));

        if (status != nullptr) {
            for (int i = 0; i < ctx.letterbox_params.size(); ++i) {
                ctx.promises_[i]->set_exception(0);
            }

            // ⭐ 使用 C API 获取错误信息
            const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
            if (api != nullptr) {
                const char* msg = api->GetErrorMessage(status);
                LOG_ERROR("ONNX Runtime error: {}", msg);
            } else {
                LOG_ERROR("ONNX Runtime error occurred, but cannot get error message");
            }
            return;
        }
        cudaEventRecord(self.compute_done_.get(), self.compute_stream_.get());

        // d2h 流需要等待事件self.compute_done_ 完成才可以开始d2h工作
        cudaStreamWaitEvent(ctx.d2h_stream.get(), self.compute_done_.get());

        const int batch = ctx.letterbox_params.size();
        
        // postprocess filter kernel
        lunchPostprocessFilter(
            ctx.d_buffer.g_output.get(), 
            ctx.d_output.get(),
            ctx.d_counts.get(),
            self.context_.output_plane_size,
            batch,
            self.context_.num_predictions,
            self.context_.num_classes,
            self.context_.conf_threshold,
            self.context_.max_detections,
            ctx.d2h_stream.get()
        );

        // ========== Step 2: D2H 拷贝计数 ==========
        cudaMemcpyAsync(
            ctx.h_counts.get(), 
            ctx.d_counts.get(), 
            batch * sizeof(int), 
            cudaMemcpyDeviceToHost, 
            ctx.d2h_stream.get()
        );
        cudaStreamSynchronize(ctx.d2h_stream.get());
        LOG_DEBUG("h_counts[0]={}", ctx.h_counts[0]);

        // 钳制并计算总检测数
        // int total_dets = 0;
        // for (int i = 0; i < batch; ++i) {
        //     if (ctx.h_counts[i] <= 0 || ctx.h_counts[i] > self.context_.max_detections) {
        //         ctx.h_counts[i] = std::clamp(ctx.h_counts[i], 0, static_cast<int> (self.context_.max_detections));
        //     }
        //     total_dets += ctx.h_counts[i];
        // }

        size_t oupt_plane_size = self.context_.max_detections * 6;
        ctx.output_plane_size = oupt_plane_size;

        // ========== Step 3: D2H 拷贝紧凑数据 ==========
        cudaMemcpyAsync(
            ctx.h_output.get(), 
            ctx.d_output.get(),
            batch * oupt_plane_size * sizeof(float),
            cudaMemcpyDeviceToHost, 
            ctx.d2h_stream.get()
        );
        
        // ✅ Step 3: 提交到后处理线程池（不阻塞回调线程）
        self.postpreprocess_pool_.enqueue([&self, &ctx]() {
            // 仅阻塞当前后处理线程，等待本次 D2H 完成
            CUDA_CHECK(cudaStreamSynchronize(ctx.d2h_stream.get()));

            assert(ctx.promises_.size() == ctx.letterbox_params.size());
            
            // 现在 host_output_buffer_ 数据安全可用
            std::vector<Detector::OutputType> dets;

            bool is_err = false;
            try {
                dets = self.postprocess(ctx);
            } catch(...)
            {
                is_err = true;
                LOG_ERROR("postprocess error");
                for (int i = 0; i < ctx.letterbox_params.size(); ++i) {
                    ctx.promises_[i]->set_exception(std::current_exception());
                }
            }
            
            if (!is_err) {
                for (int i = 0; i < dets.size(); ++i) {
                    ctx.promises_[i]->set_value(std::move(dets[i]));
                }
            }

            // 回收资源
            self.ctx_id_que_.push(ctx.ctx_id);
        });
    }


#define LUNCH_BATCH_PREPROCESS(max_batch)   \
    launchBatchPreprocess<max_batch>(   \
        ctx.d_rgb_srcs.get(),   \
        ctx.d_img_offsets.get(),    \
        ctx.d_metas.get(),  \
        batch,  \
        ctx.d_buffer.g_input.get(), \
        context_.img_size.height,   \
        context_.img_size.width,    \
        context_.norm_scale,    \
        context_.pad_value, \
        ctx.h2d_stream.get()    \
    );  \

    /// @brief gpu推理的部分
    /// @param imgs 推理的原图
    /// @param ctx 缓存数据
    void gpu_process(const std::vector<cv::Mat>& imgs, InferRequestContext& ctx) {
        // 这里的所有过程异步
        int batch = imgs.size();
        assert(batch <= context_.batch_size && "batch infer mini batch must <= batch_size");
        
        size_t offset = 0;

        for (int i = 0; i < batch; ++i) {
            const auto& img = imgs[i];

            // CPU计算好参数将数据传入GPU
            cudaMemcpyAsync(
                ctx.d_rgb_srcs.get() + offset, 
                img.data, 
                img.rows * img.step, 
                cudaMemcpyHostToDevice,
                ctx.h2d_stream.get()
            );

            ctx.h_img_offsets[i] = offset;
            LOG_TRACE("offset: {}", offset);

            offset += (img.rows * img.step); // 修改偏移量
        }

        LOG_TRACE("MAX POSITION: {}", context_.batch_size * kMaxImage);

        // 拷贝offsets 到gpu
        cudaMemcpyAsync(
            ctx.d_img_offsets.get(), 
            ctx.h_img_offsets.get(), 
            batch * sizeof(size_t), 
            cudaMemcpyHostToDevice,
            ctx.h2d_stream.get()
        );

        cudaMemcpyAsync(
            ctx.d_metas.get(), 
            ctx.h_metas.get(),
            batch * sizeof(ImageMeta), 
            cudaMemcpyHostToDevice,
            ctx.h2d_stream.get()
        );

        // 启用gpu预处理kernel
        if (context_.batch_size <= 1) {
            LUNCH_BATCH_PREPROCESS(1);
        } else if(context_.batch_size <= 2) {
            LUNCH_BATCH_PREPROCESS(2);
        } else if(context_.batch_size <= 4) {
            LUNCH_BATCH_PREPROCESS(4);
        } else if(context_.batch_size <= 8) {
            LUNCH_BATCH_PREPROCESS(8);
        } else if(context_.batch_size <= 16) {
            LUNCH_BATCH_PREPROCESS(16);
        } else {
            throw std::runtime_error(MESSAGE_WITH_LOC("Batch size exceed 16 !"));
        }

        cudaEventRecord(ctx.h2d_done.get(), ctx.h2d_stream.get());

        // h2d 完成才可以开始推理
        cudaStreamWaitEvent(compute_stream_.get(), ctx.h2d_done.get());

        // kernel
        context_.session->RunAsync(
            context_.run_options,
            context_.input_names_c_str.data(),
            &ctx.d_buffer.input_tensor,
            context_.input_names_c_str.size(),
            context_.output_names_c_str.data(), 
            &ctx.d_buffer.output_tensor, 
            context_.output_names_c_str.size(),
            callback_func,
            &ctx
        );
    }

    /// @brief CPU上后处理的过程
    /// @param ctx 当前所使用的缓存上下文
    /// @return
    std::vector<std::vector<Detection>> postprocess(const InferRequestContext& ctx) {
        LOG_DEBUG("postprocess call");
        int batch = ctx.letterbox_params.size();

        LOG_DEBUG("actual batch size: {}", batch);

        assert(batch && "Infer batch must > 0");

        size_t oupt_plane_size = context_.max_detections * 6;

        // ========== Step 4: CPU 解码 + NMS ==========
        std::vector<std::vector<Detection>> results(batch);

        LOG_DEBUG("h_counts0 = {}", ctx.h_counts[0]);

        for (size_t b = 0; b < batch; ++b) {
            const int cnt = ctx.h_counts[b];
            if (cnt == 0) continue;

            // ⚠️ 关键：按 MAX_GPU_DETECTIONS 寻址，不是按 host_counts
            const float* pdata = ctx.h_output.get() + b * ctx.output_plane_size;

            std::vector<cv::Rect2i> boxes;
            std::vector<float> confidences;
            std::vector<int> class_ids;
            boxes.reserve(cnt);
            confidences.reserve(cnt);
            class_ids.reserve(cnt);

            const auto& lp = ctx.letterbox_params[b];

            for (int i = 0; i < cnt; ++i) {
                const float* det = pdata + i * 6;
                float cx = det[0], cy = det[1], w = det[2], h = det[3];
                float score = det[4];
                int cls = static_cast<int>(det[5]);

                float x1 = std::clamp((cx - 0.5f*w - lp.pad_left) / lp.scale, 0.f, (float)lp.orig_w);
                float y1 = std::clamp((cy - 0.5f*h - lp.pad_top)  / lp.scale, 0.f, (float)lp.orig_h);
                float x2 = std::clamp((cx + 0.5f*w - lp.pad_left) / lp.scale, 0.f, (float)lp.orig_w);
                float y2 = std::clamp((cy + 0.5f*h - lp.pad_top)  / lp.scale, 0.f, (float)lp.orig_h);

                boxes.emplace_back(cv::Rect2i(
                    static_cast<int>(x1), static_cast<int>(y1),
                    static_cast<int>(x2 - x1), static_cast<int>(y2 - y1)));
                confidences.push_back(score);
                class_ids.push_back(cls);
            }

            // NMS
            std::vector<int> nms_indices;
            cv::dnn::NMSBoxes(boxes, confidences, context_.conf_threshold, context_.nms_threshold,nms_indices);

            std::vector<Detection>& batch_result = results[b];
            batch_result.reserve(nms_indices.size());
            for (int idx : nms_indices) {
                Detection d;
                d.box = boxes[idx];
                d.score = confidences[idx];
                d.class_id = class_ids[idx];
                batch_result.push_back(d);
            }
        }

        return results;
    }

    /// @brief 异步提交任务至流水线
    /// @param img 源图像
    /// @return future
    std::future<Detector::OutputType> submit(const cv::Mat& img) {
        std::shared_ptr<std::promise<Detector::OutputType>> spm = std::make_shared<std::promise<Detector::OutputType>>();

        std::future<Detector::OutputType> fut = spm->get_future();

        // 直接将任务提交至CPU预处理队列
        // 预处理完成后将数据放入infer队列
        preprocess_pool_.enqueue([this, spm, img](){
            {
                std::lock_guard<std::mutex> lock(mutex_);
                img_que_.push(img);
                img_meta_que_.push(preprocess_v1(img));
                promises_.push(spm);
            }
            condition_.notify_one();
        });

        return fut;
    }
};