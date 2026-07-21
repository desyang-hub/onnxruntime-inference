#include "benchmark_runner.h"

#include <filesystem>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <chrono>

#ifdef _WIN32
#include <io.h>   // _popen / _pclose
#else
#include <cstdio> // popen / pclose
#endif

#include <yaml-cpp/yaml.h>

namespace fs = std::filesystem;

// ============================================================
// GpuResourceSampler
// ============================================================

void GpuResourceSampler::start(int gpu_id) {
    gpu_id_ = gpu_id;
    running_.store(true);

    worker_ = std::thread([this]() {
        constexpr int interval_ms = 200;
        while (running_.load()) {
            std::string cmd =
                "nvidia-smi --query-gpu=utilization.gpu,memory.used "
                "--format=csv,nounits,noheader -i " +
                std::to_string(gpu_id_);

#ifdef _WIN32
            FILE* pipe = _popen(cmd.c_str(), "r");
#else
            FILE* pipe = popen(cmd.c_str(), "r");
#endif
            if (pipe) {
                char buf[256] = {0};
                if (fgets(buf, sizeof(buf), pipe)) {
                    std::string line(buf);
                    auto comma_pos = line.find(',');
                    if (comma_pos != std::string::npos) {
                        try {
                            double util = std::stod(line.substr(0, comma_pos));
                            double mem  = std::stod(line.substr(comma_pos + 1));
                            std::lock_guard<std::mutex> lock(mutex_);
                            utilizations_.push_back(util);
                            mem_usages_.push_back(mem);
                        } catch (...) {}
                    }
                }
#ifdef _WIN32
                _pclose(pipe);
#else
                pclose(pipe);
#endif
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
        }
    });
}

std::pair<double, double> GpuResourceSampler::stop() {
    running_.store(false);
    if (worker_.has_value() && worker_->joinable()) {
        worker_->join();
    }

    double avg_util = 0.0, avg_mem = 0.0;
    if (!utilizations_.empty()) {
        avg_util = std::accumulate(utilizations_.begin(), utilizations_.end(), 0.0)
                   / utilizations_.size();
        avg_mem  = std::accumulate(mem_usages_.begin(), mem_usages_.end(), 0.0)
                   / mem_usages_.size();
    }
    return {avg_util, avg_mem};
}

// ============================================================
// ResultReporter
// ============================================================

void ResultReporter::print_table(const std::vector<BenchmarkResult>& results) {
    std::cout << "\n";
    std::cout << "╔══════════╦═════════╦════════════╦══════════╦══════════╦══════════╦══════════╦══════════╗\n";
    std::cout << "║   Mode   ║  Reqs   ║ Wall Time  ║  Through ║   P50    ║   P90    ║   P95    ║   P99    ║\n";
    std::cout << "║          ║         ║    (ms)    ║   (QPS)  ║   (ms)   ║   (ms)   ║   (ms)   ║   (ms)   ║\n";
    std::cout << "╠══════════╬═════════╬════════════╬══════════╬══════════╬══════════╬══════════╬══════════╣\n";

    for (const auto& r : results) {
        std::cout << "║ "
                  << std::left  << std::setw(8) << r.mode_name << " ║ "
                  << std::right << std::setw(7) << r.total_requests << " ║ "
                  << std::setw(10) << std::fixed << std::setprecision(1) << r.total_wall_ms << " ║ "
                  << std::setw(8) << std::fixed << std::setprecision(1) << r.throughput_fps << " ║ "
                  << std::setw(8) << std::fixed << std::setprecision(2) << r.latency_p50_ms << " ║ "
                  << std::setw(8) << std::fixed << std::setprecision(2) << r.latency_p90_ms << " ║ "
                  << std::setw(8) << std::fixed << std::setprecision(2) << r.latency_p95_ms << " ║ "
                  << std::setw(8) << std::fixed << std::setprecision(2) << r.latency_p99_ms << " ║\n";
    }

    std::cout << "╠══════════╬═════════╬════════════╬══════════╬══════════╬══════════╬══════════╬══════════╣\n";
    std::cout << "║   Mode   ║   Avg   ║    Min     ║    Max   ║\n";
    std::cout << "║          ║   (ms)  ║    (ms)    ║    (ms)  ║\n";
    std::cout << "╠══════════╬═════════╬════════════╬══════════╣\n";

    for (const auto& r : results) {
        std::cout << "║ "
                  << std::left  << std::setw(8) << r.mode_name << " ║ "
                  << std::setw(7) << std::fixed << std::setprecision(2) << r.latency_avg_ms << " ║ "
                  << std::setw(10) << std::fixed << std::setprecision(2) << r.latency_min_ms << " ║ "
                  << std::setw(8) << std::fixed << std::setprecision(2) << r.latency_max_ms << " ║\n";
    }

    std::cout << "╚══════════╩═════════╩════════════╩══════════╩══════════╩══════════╩══════════╩══════════╝\n";
    std::cout << "\n";
}

void ResultReporter::export_json(const std::vector<BenchmarkResult>& results,
                                  const std::string& filepath) {
    std::ofstream ofs(filepath);
    if (!ofs.is_open()) {
        LOG_ERROR("Cannot open JSON file: {}", filepath);
        return;
    }

    ofs << "{\n  \"timestamp\": \""
        << std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch()).count()
        << "\",\n  \"results\": [\n";

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        ofs << "    {\n";
        ofs << "      \"mode\": \"" << r.mode_name << "\",\n";
        ofs << "      \"total_requests\": " << r.total_requests << ",\n";
        ofs << "      \"total_wall_ms\": " << std::fixed << std::setprecision(2)
            << r.total_wall_ms << ",\n";
        ofs << "      \"throughput_qps\": " << std::fixed << std::setprecision(2)
            << r.throughput_fps << ",\n";
        ofs << "      \"latency_avg_ms\": " << std::fixed << std::setprecision(4)
            << r.latency_avg_ms << ",\n";
        ofs << "      \"latency_min_ms\": " << std::fixed << std::setprecision(4)
            << r.latency_min_ms << ",\n";
        ofs << "      \"latency_max_ms\": " << std::fixed << std::setprecision(4)
            << r.latency_max_ms << ",\n";
        ofs << "      \"latency_p50_ms\": " << std::fixed << std::setprecision(4)
            << r.latency_p50_ms << ",\n";
        ofs << "      \"latency_p90_ms\": " << std::fixed << std::setprecision(4)
            << r.latency_p90_ms << ",\n";
        ofs << "      \"latency_p95_ms\": " << std::fixed << std::setprecision(4)
            << r.latency_p95_ms << ",\n";
        ofs << "      \"latency_p99_ms\": " << std::fixed << std::setprecision(4)
            << r.latency_p99_ms << ",\n";
        ofs << "      \"samples_count\": " << r.latency_samples_ms.size() << "\n";
        ofs << "    }";
        if (i + 1 < results.size()) ofs << ",";
        ofs << "\n";
    }

    ofs << "  ]\n}\n";
    LOG_INFO("JSON report exported to: {}", filepath);
}

void ResultReporter::export_csv(const std::vector<BenchmarkResult>& results,
                                 const std::string& filepath) {
    std::ofstream ofs(filepath);
    if (!ofs.is_open()) {
        LOG_ERROR("Cannot open CSV file: {}", filepath);
        return;
    }

    ofs << "mode,total_requests,wall_time_ms,throughput_qps,"
        << "latency_avg_ms,latency_min_ms,latency_max_ms,"
        << "latency_p50_ms,latency_p90_ms,latency_p95_ms,latency_p99_ms\n";

    for (const auto& r : results) {
        ofs << r.mode_name << ","
            << r.total_requests << ","
            << std::fixed << std::setprecision(2) << r.total_wall_ms << ","
            << std::fixed << std::setprecision(2) << r.throughput_fps << ","
            << std::fixed << std::setprecision(4) << r.latency_avg_ms << ","
            << std::fixed << std::setprecision(4) << r.latency_min_ms << ","
            << std::fixed << std::setprecision(4) << r.latency_max_ms << ","
            << std::fixed << std::setprecision(4) << r.latency_p50_ms << ","
            << std::fixed << std::setprecision(4) << r.latency_p90_ms << ","
            << std::fixed << std::setprecision(4) << r.latency_p95_ms << ","
            << std::fixed << std::setprecision(4) << r.latency_p99_ms << "\n";
    }

    LOG_INFO("CSV report exported to: {}", filepath);
}

// ============================================================
// Config parser from YAML
// ============================================================

static ModelConfigOverride parse_override(const YAML::Node& node) {
    if (!node) return {};
    ModelConfigOverride ov;
    ov.batch = node["batch"].as<size_t>(0);
    ov.buffer_size = node["buffer_size"].as<size_t>(0);
    ov.warm_up = node["warm_up"].as<int>(-1);
    ov.execution_providers = node["execution_providers"].as<std::string>("");
    return ov;
}

BenchmarkConfig parse_benchmark_config(const std::string& yaml_path) {
    YAML::Node cfg = YAML::LoadFile(yaml_path);
    BenchmarkConfig bc;

    bc.config_path        = cfg["config_path"].as<std::string>(yaml_path);
    bc.image_path         = cfg["image_path"].as<std::string>("assets/bus.png");
    bc.warmup_count       = cfg["warmup_count"].as<size_t>(5);
    bc.test_duration_sec  = cfg["test_duration_sec"].as<double>(10.0);
    bc.min_iterations     = cfg["min_iterations"].as<size_t>(50);
    bc.concurrent_clients = cfg["concurrent_clients"].as<int>(1);
    bc.output_dir         = cfg["output_dir"].as<std::string>("./benchmark_output");
    bc.export_json        = cfg["export_json"].as<bool>(true);
    bc.export_csv         = cfg["export_csv"].as<bool>(true);

    // Per-scheduler overrides
    bc.sync_override  = parse_override(cfg["sync_override"]);
    bc.async_override = parse_override(cfg["async_override"]);
    bc.batch_override = parse_override(cfg["batch_override"]);

    // Runtime options
    bc.mode = cfg["mode"].as<std::string>("all");
    bc.model_type = cfg["model_type"].as<std::string>("yolo");
    bc.show_gpu_stats = cfg["show_gpu_stats"].as<bool>(true);

    return bc;
}
