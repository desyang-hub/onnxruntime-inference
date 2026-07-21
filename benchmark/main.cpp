/**
 * @file main.cpp
 * @brief CLI entry point for the benchmark tool
 *
 * Usage:
 *   ./benchmark --config config/model_config.yaml --image assets/bus.png
 *               --warmup 5 --duration 10 --clients 1 --mode all
 *               --output-dir ./benchmark_output
 *
 * Modes: sync | async | batch | all
 *   - sync:   SyncScheduler (sequential)
 *   - async:  AsyncScheduler (pipelined)
 *   - batch:  BatchScheduler (micro-batch, GPU only)
 *   - all:    Run all three and compare
 */

#include <iostream>
#include <string>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <cstdlib>

#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>

#include "logger/logger.h"
#include "benchmark_runner.h"

// Include runner headers
#include "runner/detect/YoloDetector.h"
#include "runner/image_restoration/NAFNet.h"

#include "scheduler/SyncScheduler.h"
#include "scheduler/AsyncScheduler.h"

#ifdef ENABLE_CUDA
#include "scheduler/BatchScheduler.h"
#endif

namespace fs = std::filesystem;

// ============================================================
// CLI argument parser (no external dependency)
// ============================================================

struct CliArgs {
    std::string config_path = "config/model_config.yaml";
    std::string image_path = "assets/bus.png";
    size_t warmup_count = 5;
    double test_duration_sec = 10.0;
    size_t min_iterations = 50;
    int concurrent_clients = 1;
    std::string mode = "all";
    std::string output_dir = "./benchmark_output";
    std::string model_type = "yolo";   // yolo | nafnet
    bool export_json = true;
    bool export_csv = true;
    bool show_gpu_stats = true;
};

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "\nOptions:\n"
              << "  --config PATH         Model config YAML path\n"
              << "  --image PATH          Input image path for benchmark\n"
              << "  --warmup N            Warmup iterations (default: 5)\n"
              << "  --duration SEC        Steady-state test duration in seconds (default: 10)\n"
              << "  --min-iterations N    Minimum iterations regardless of duration (default: 50)\n"
              << "  --clients N           Concurrent submission threads (default: 1)\n"
              << "  --mode MODE           sync | async | batch | all (default: all)\n"
              << "  --output-dir DIR      Output directory for reports (default: ./benchmark_output)\n"
              << "  --model TYPE          yolo | nafnet (default: yolo)\n"
              << "  --no-json             Skip JSON export\n"
              << "  --no-csv              Skip CSV export\n"
              << "  --no-gpu-stats        Skip GPU resource sampling\n"
              << "  --help                Show this help\n";
}

static CliArgs parse_args(int argc, char* argv[]) {
    CliArgs args;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        auto get_next = [&]() -> std::string {
            if (i + 1 < argc) return argv[++i];
            throw std::runtime_error(std::string("Missing value for ") + arg);
        };

        if (arg == "--config")         args.config_path = get_next();
        else if (arg == "--image")     args.image_path = get_next();
        else if (arg == "--warmup")    args.warmup_count = std::stoull(get_next());
        else if (arg == "--duration")  args.test_duration_sec = std::stod(get_next());
        else if (arg == "--min-iterations") args.min_iterations = std::stoull(get_next());
        else if (arg == "--clients")   args.concurrent_clients = std::stoi(get_next());
        else if (arg == "--mode")      args.mode = get_next();
        else if (arg == "--output-dir") args.output_dir = get_next();
        else if (arg == "--model")     args.model_type = get_next();
        else if (arg == "--no-json")   args.export_json = false;
        else if (arg == "--no-csv")    args.export_csv = false;
        else if (arg == "--no-gpu-stats") args.show_gpu_stats = false;
        else if (arg == "--help")      { print_usage(argv[0]); std::exit(0); }
        else throw std::runtime_error("Unknown argument: " + arg);
    }

    if (args.config_path.empty())
        throw std::runtime_error("--config is required. Use --help for usage.");
    if (args.mode != "sync" && args.mode != "async" && args.mode != "batch" && args.mode != "all")
        throw std::runtime_error("--mode must be one of: sync, async, batch, all");

    return args;
}

// ============================================================
// Benchmark execution — dispatch based on model type + mode
// ============================================================

template <class Runner>
std::vector<BenchmarkResult> execute_benchmark(
    std::shared_ptr<Runner> runner,
    const BenchmarkConfig& config,
    const CliArgs& cli)
{
    BenchmarkRunner<Runner> bench;

#ifdef ENABLE_CUDA
    if (cli.show_gpu_stats) {
        bench.set_gpu_sampler(std::make_unique<GpuResourceSampler>());
    }
#endif

    std::vector<BenchmarkResult> results;

    if (cli.mode == "sync" || cli.mode == "all") {
        std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        std::cout << "  Running SyncScheduler benchmark...\n";
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        results.push_back(bench.run_sync(runner, config));
    }

    if (cli.mode == "async" || cli.mode == "all") {
        std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        std::cout << "  Running AsyncScheduler benchmark...\n";
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        results.push_back(bench.run_async(runner, config));
    }

#ifdef ENABLE_CUDA
    if (cli.mode == "batch" || cli.mode == "all") {
        std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        std::cout << "  Running BatchScheduler benchmark...\n";
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        results.push_back(bench.run_batch(runner, config));
    }
#else
    if (cli.mode == "batch") {
        std::cerr << "BatchScheduler requires CUDA. Rebuild with -DENABLE_CUDA=ON\n";
        std::exit(1);
    }
#endif

    return results;
}

// ============================================================
// Main
// ============================================================

int main(int argc, char* argv[]) {
    try {
        CliArgs cli = parse_args(argc, argv);

        // Init logger — suppress noisy logs during benchmark
        logger::Init(logger::LOGLEVEL_WARN);

        std::cout << "╔══════════════════════════════════════════════════════════╗\n";
        std::cout << "║         ONNXRuntime Inference Benchmark Tool             ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════╝\n";

        std::cout << "\nConfiguration:\n";
        std::cout << "  Config path:      " << cli.config_path << "\n";
        std::cout << "  Image path:       " << cli.image_path << "\n";
        std::cout << "  Model type:       " << cli.model_type << "\n";
        std::cout << "  Mode:             " << cli.mode << "\n";
        std::cout << "  Warmup:           " << cli.warmup_count << " iterations\n";
        std::cout << "  Duration:         " << cli.test_duration_sec << " seconds\n";
        std::cout << "  Min iterations:   " << cli.min_iterations << "\n";
        std::cout << "  Concurrent:       " << cli.concurrent_clients << " client(s)\n";

        // Build BenchmarkConfig
        BenchmarkConfig config;
        config.config_path        = cli.config_path;
        config.image_path         = cli.image_path;
        config.warmup_count       = cli.warmup_count;
        config.test_duration_sec  = cli.test_duration_sec;
        config.min_iterations     = cli.min_iterations;
        config.concurrent_clients = cli.concurrent_clients;
        config.output_dir         = cli.output_dir;
        config.export_json        = cli.export_json;
        config.export_csv         = cli.export_csv;

        // Verify inputs
        if (!fs::exists(config.image_path)) {
            std::cerr << "Error: Image not found: " << config.image_path << "\n";
            return 1;
        }
        if (!fs::exists(config.config_path)) {
            std::cerr << "Error: Config not found: " << config.config_path << "\n";
            return 1;
        }

        std::vector<BenchmarkResult> results;

        if (cli.model_type == "yolo") {
            auto runner = std::make_shared<YoloDetector>(YAML::LoadFile(config.config_path));
            results = execute_benchmark<YoloDetector>(runner, config, cli);
        }
        else if (cli.model_type == "nafnet") {
            auto runner = std::make_shared<NAFNet>(YAML::LoadFile(config.config_path));
            results = execute_benchmark<NAFNet>(runner, config, cli);
        }
        else {
            std::cerr << "Unknown model type: " << cli.model_type
                      << ". Supported: yolo, nafnet\n";
            return 1;
        }

        // Print results table
        ResultReporter::print_table(results);

        // Export
        fs::create_directories(config.output_dir);

        std::string timestamp;
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        {
            std::tm tm_buf;
#ifdef _WIN32
            localtime_s(&tm_buf, &time_t);
#else
            localtime_r(&time_t, &tm_buf);
#endif
            std::ostringstream oss;
            oss << std::put_time(&tm_buf, "%Y%m%d_%H%M%S");
            timestamp = oss.str();
        }

        if (config.export_json) {
            std::string json_path = config.output_dir + "/benchmark_" + timestamp + ".json";
            ResultReporter::export_json(results, json_path);
        }
        if (config.export_csv) {
            std::string csv_path = config.output_dir + "/benchmark_" + timestamp + ".csv";
            ResultReporter::export_csv(results, csv_path);
        }

        std::cout << "Benchmark complete.\n";
        return 0;

    }
    catch (const std::exception& e) {
        std::cerr << "Benchmark error: " << e.what() << "\n";
        return 1;
    }
}
