#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <memory>
#include <functional>
#include <future>
#include <thread>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <optional>

#include <opencv2/opencv.hpp>

#include "scheduler/SyncScheduler.h"
#include "scheduler/AsyncScheduler.h"
#include "scheduler/type_trait/runner_type_trait.h"
#include "logger/logger.h"
#include "device/cuda_utils.h"

#ifdef ENABLE_CUDA
#include "scheduler/BatchScheduler.h"
#endif

// ============================================================
// Configuration
// ============================================================

// Per-scheduler model config override
// Only the fields you set will override the base model config;
// the rest are inherited from the base YAML.
struct ModelConfigOverride {
    size_t batch = 0;             // 0 = use base config value
    size_t buffer_size = 0;       // 0 = use base config value
    int warm_up = -1;             // -1 = use base config value
    std::string execution_providers; // empty = use base config value
};

struct BenchmarkConfig {
    std::string config_path;
    std::string image_path;

    size_t warmup_count = 5;
    double test_duration_sec = 10.0;
    size_t min_iterations = 50;

    int concurrent_clients = 1;

    /// Per-scheduler model overrides — each mode gets its own runner
    /// with the override applied on top of the base config.
    ModelConfigOverride sync_override;
    ModelConfigOverride async_override;
    ModelConfigOverride batch_override;

    /// Runtime options
    std::string mode = "all";
    std::string model_type = "yolo";
    bool show_gpu_stats = true;

    std::string output_dir = "./benchmark_output";
    bool export_json = true;
    bool export_csv = true;
};

// ============================================================
// Benchmark Result for one scheduling mode
// ============================================================

struct BenchmarkResult {
    std::string mode_name;
    size_t total_requests = 0;
    double total_wall_ms = 0.0;

    double throughput_fps = 0.0;
    double latency_p50_ms = 0.0;
    double latency_p90_ms = 0.0;
    double latency_p95_ms = 0.0;
    double latency_p99_ms = 0.0;
    double latency_avg_ms = 0.0;
    double latency_min_ms = 0.0;
    double latency_max_ms = 0.0;

    std::vector<double> latency_samples_ms;
};

// ============================================================
// GPU resource sampler (best-effort, via nvidia-smi subprocess)
// ============================================================

class GpuResourceSampler {
public:
    GpuResourceSampler() = default;
    ~GpuResourceSampler() = default;

    void start(int gpu_id = 0);
    std::pair<double, double> stop();

private:
    std::optional<std::thread> worker_;
    std::atomic<bool> running_{false};
    std::mutex mutex_;
    std::vector<double> utilizations_;
    std::vector<double> mem_usages_;
    int gpu_id_ = 0;
};

// ============================================================
// Result Reporter
// ============================================================

class ResultReporter {
public:
    static void print_table(const std::vector<BenchmarkResult>& results);
    static void export_json(const std::vector<BenchmarkResult>& results,
                            const std::string& filepath);
    static void export_csv(const std::vector<BenchmarkResult>& results,
                           const std::string& filepath);
};

// ============================================================
// Config parser from YAML
// ============================================================

BenchmarkConfig parse_benchmark_config(const std::string& yaml_path);

// ============================================================
// Internal helpers
// ============================================================

namespace detail {

inline double percentile(std::vector<double>& samples, double p) {
    if (samples.empty()) return 0.0;
    std::sort(samples.begin(), samples.end());
    double idx = p / 100.0 * (samples.size() - 1);
    size_t lo  = static_cast<size_t>(std::floor(idx));
    size_t hi  = static_cast<size_t>(std::ceil(idx));
    if (lo == hi || hi >= samples.size()) return samples[lo];
    double frac = idx - lo;
    return samples[lo] * (1.0 - frac) + samples[hi] * frac;
}

inline BenchmarkResult compute_result(const std::string& mode_name,
                                      std::vector<double> samples_ms,
                                      double total_wall_ms) {
    BenchmarkResult result;
    result.mode_name = mode_name;
    result.total_requests = samples_ms.size();
    result.total_wall_ms  = total_wall_ms;
    result.latency_samples_ms = samples_ms;

    if (samples_ms.empty()) return result;

    result.latency_min_ms = *std::min_element(samples_ms.begin(), samples_ms.end());
    result.latency_max_ms = *std::max_element(samples_ms.begin(), samples_ms.end());
    result.latency_avg_ms = std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0)
                            / samples_ms.size();
    result.latency_p50_ms = percentile(samples_ms, 50.0);
    result.latency_p90_ms = percentile(samples_ms, 90.0);
    result.latency_p95_ms = percentile(samples_ms, 95.0);
    result.latency_p99_ms = percentile(samples_ms, 99.0);

    result.throughput_fps = (total_wall_ms > 0)
        ? (samples_ms.size() / total_wall_ms * 1000.0)
        : 0.0;

    return result;
}

} // namespace detail

// ============================================================
// BenchmarkRunner — full template implementation in header
// ============================================================

template <class Runner>
class BenchmarkRunner {
private:
    using InputType  = typename runner_type_trait<Runner>::InputType;
    using OutputType = typename runner_type_trait<Runner>::OutputType;

    std::unique_ptr<GpuResourceSampler> gpu_sampler_ = nullptr;
    using Clock = std::chrono::high_resolution_clock;

    // --------------------------------------------------------
    // Internal: apply ModelConfigOverride on top of a YAML::Node
    // Returns a new YAML::Node with the overrides applied.
    // --------------------------------------------------------
    static YAML::Node apply_override(const YAML::Node& base,
                                     const ModelConfigOverride& ov)
    {
        YAML::Node node = base; // deep copy
        if (ov.batch > 0)
            node["model"]["batch"] = ov.batch;
        if (ov.buffer_size > 0)
            node["model"]["buffer_size"] = ov.buffer_size;
        if (ov.warm_up >= 0)
            node["model"]["warm_up"] = ov.warm_up;
        if (!ov.execution_providers.empty())
            node["model"]["session_options"]["execution_providers"] = ov.execution_providers;
        return node;
    }

    // --------------------------------------------------------
    // Internal: warmup + steady-state for any submit function
    // --------------------------------------------------------
    template <typename SubmitFn>
    BenchmarkResult run_pipeline(const std::string& mode_name,
                                  const cv::Mat& test_img,
                                  const BenchmarkConfig& config,
                                  SubmitFn submit_fn)
    {
        // --- Warmup ---
        {
            LOG_INFO("[Warmup] Running {} {} iterations...", mode_name, config.warmup_count);

#ifdef ENABLE_CUDA
            cudaEvent_t warmup_start, warmup_end;
            cudaEventCreate(&warmup_start);
            cudaEventCreate(&warmup_end);
            cudaEventRecord(warmup_start);
#endif

            for (size_t i = 0; i < config.warmup_count; ++i) {
                auto fut = submit_fn(test_img);
                fut.get();
            }

#ifdef ENABLE_CUDA
            cudaEventRecord(warmup_end);
            cudaEventSynchronize(warmup_end);
            float warmup_ms = 0;
            cudaEventElapsedTime(&warmup_ms, warmup_start, warmup_end);
            LOG_INFO("[Warmup] {} completed in {:.1f} ms", mode_name, warmup_ms);
            cudaEventDestroy(warmup_start);
            cudaEventDestroy(warmup_end);
#else
            LOG_INFO("[Warmup] {} completed", mode_name);
#endif
        }

        // --- Steady-state ---
        LOG_INFO("[Benchmark] Running {} steady-state test (duration={:.1f}s)...",
                 mode_name, config.test_duration_sec);

        std::vector<double> latencies_ms;
        std::mutex latencies_mutex;

        auto wall_start = Clock::now();

        auto submit_and_measure = [&]() {
            // cv::Mat local_img = test_img.clone();
            std::vector<double> local_latencies;

            while (true) {
                auto now = Clock::now();
                double elapsed = std::chrono::duration<double>(now - wall_start).count();
                if (elapsed >= config.test_duration_sec &&
                    static_cast<size_t>(local_latencies.size()) >= config.min_iterations) {
                    break;
                }

                auto t0 = Clock::now();
                auto fut = submit_fn(test_img);
                fut.get();
                auto t1 = Clock::now();

                double lat_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                local_latencies.push_back(lat_ms);
            }

            std::lock_guard<std::mutex> lock(latencies_mutex);
            latencies_ms.insert(latencies_ms.end(),
                                local_latencies.begin(), local_latencies.end());
        };

        std::vector<std::thread> clients;
        clients.reserve(config.concurrent_clients);
        for (int i = 0; i < config.concurrent_clients; ++i) {
            clients.emplace_back(submit_and_measure);
        }
        for (auto& t : clients) t.join();

        auto wall_end = Clock::now();
        double total_wall_ms =
            std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

#ifdef ENABLE_CUDA
        cudaDeviceSynchronize();
#endif

        LOG_INFO("[Benchmark] {}: {} requests in {:.1f} ms",
                 mode_name, latencies_ms.size(), total_wall_ms);
        return detail::compute_result(mode_name, std::move(latencies_ms), total_wall_ms);
    }

public:
    BenchmarkRunner() = default;
    ~BenchmarkRunner() = default;
    BenchmarkRunner(const BenchmarkRunner&) = delete;
    BenchmarkRunner& operator=(const BenchmarkRunner&) = delete;

    void set_gpu_sampler(std::unique_ptr<GpuResourceSampler> sampler) {
        gpu_sampler_ = std::move(sampler);
    }

    // --------------------------------------------------------
    // Public API — each mode creates its own runner with the
    // corresponding config override applied.
    // --------------------------------------------------------

    BenchmarkResult run_sync(const YAML::Node& base_config,
                             const BenchmarkConfig& config)
    {
        auto test_img = cv::imread(config.image_path);
        if (test_img.empty()) {
            throw std::runtime_error("Cannot load benchmark image: " + config.image_path);
        }

        YAML::Node node = apply_override(base_config, config.sync_override);
        std::shared_ptr<Runner> runner = std::make_shared<Runner>(node);
        SyncScheduler<Runner> scheduler(runner);

        auto submit_fn = [&](const cv::Mat& img) {
            return scheduler.submit(img);
        };
        return run_pipeline("Sync", test_img, config, std::move(submit_fn));
    }

    BenchmarkResult run_async(const YAML::Node& base_config,
                              const BenchmarkConfig& config)
    {
        auto test_img = cv::imread(config.image_path);
        if (test_img.empty()) {
            throw std::runtime_error("Cannot load benchmark image: " + config.image_path);
        }

        YAML::Node node = apply_override(base_config, config.async_override);
        std::shared_ptr<Runner> runner = std::make_shared<Runner>(node);
        AsyncScheduler<Runner> scheduler(runner);

        auto submit_fn = [&](const cv::Mat& img) {
            return scheduler.submit(img);
        };
        return run_pipeline("Async", test_img, config, std::move(submit_fn));
    }

#ifdef ENABLE_CUDA
    BenchmarkResult run_batch(const YAML::Node& base_config,
                              const BenchmarkConfig& config)
    {
        auto test_img = cv::imread(config.image_path);
        if (test_img.empty()) {
            throw std::runtime_error("Cannot load benchmark image: " + config.image_path);
        }

        YAML::Node node = apply_override(base_config, config.batch_override);
        std::shared_ptr<Runner> runner = std::make_shared<Runner>(node);
        auto batch_scheduler = std::make_shared<BatchScheduler<Runner>>(runner);

        auto submit_fn = [&](const cv::Mat& img) {
            return batch_scheduler->submit(img);
        };
        return run_pipeline("Batch", test_img, config, std::move(submit_fn));
    }
#endif

    std::vector<BenchmarkResult> run_all(const YAML::Node& base_config,
                                         const BenchmarkConfig& config)
    {
        std::vector<BenchmarkResult> results;

        if (gpu_sampler_) gpu_sampler_->start();

        results.push_back(run_sync(base_config, config));
        results.push_back(run_async(base_config, config));

#ifdef ENABLE_CUDA
        results.push_back(run_batch(base_config, config));
#endif

        if (gpu_sampler_) {
            auto [util, mem] = gpu_sampler_->stop();
            LOG_INFO("[GPU] Avg utilization: {:.1f}%, Avg memory: {:.1f} MB", util, mem);
        }

        return results;
    }
};
