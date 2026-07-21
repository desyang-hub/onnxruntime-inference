/**
 * @file main.cpp
 * @brief CLI entry point for the benchmark tool
 *
 * All configuration comes from benchmark_config.yaml.
 * CLI flags override YAML values when provided.
 *
 * Each scheduler mode (sync/async/batch) creates its own Runner with
 * a per-scheduler config override, so the model's batch size, buffer_size,
 * etc. are optimal for each mode.
 *
 * Usage:
 *   ./benchmark --config benchmark/benchmark_config.yaml
 *   ./benchmark --config benchmark/benchmark_config.yaml --mode sync --duration 30
 *
 * benchmark_config.yaml contains config_path (model YAML), image_path,
 * warmup/duration settings, and per-scheduler overrides.
 *
 * Modes: sync | async | batch | all
 *   - sync:   SyncScheduler (sequential, batch=1)
 *   - async:  AsyncScheduler (pipelined, batch=1)
 *   - batch:  BatchScheduler (micro-batch, larger batch)
 *   - all:    Run all three and compare
 */

 #include <iostream>
 #include <string>
 #include <filesystem>
 #include <memory>
 #include <stdexcept>
 #include <cstdlib>
 #include <sstream>
 #include <iomanip>
 
 #include <yaml-cpp/yaml.h>
 #include <opencv2/opencv.hpp>
 
 #include "logger/logger.h"
 #include "benchmark_runner.h"
 
 // Include runner headers for type information
 #include "runner/detect/YoloDetector.h"
 #include "runner/image_restoration/NAFNet.h"
 
 #include "scheduler/SyncScheduler.h"
 #include "scheduler/AsyncScheduler.h"
 
 #ifdef ENABLE_CUDA
 #include "scheduler/BatchScheduler.h"
 #endif
 
 namespace fs = std::filesystem;
 
 // ============================================================
 // CLI argument parser — applies overrides directly onto BenchmarkConfig
 // ============================================================
 
 static void print_usage(const char* prog) {
     std::cout << "Usage: " << prog << " [options]\n"
               << "\nOptions:\n"
               << "  --config PATH         Benchmark config YAML (benchmark_config.yaml) [required]\n"
               << "  --image PATH          Override image path\n"
               << "  --warmup N            Override warmup iterations\n"
               << "  --duration SEC        Override test duration in seconds\n"
               << "  --min-iterations N    Override minimum iterations\n"
               << "  --clients N           Override concurrent submission threads\n"
               << "  --mode MODE           sync | async | batch | all\n"
               << "  --output-dir DIR      Override output directory\n"
               << "  --model TYPE          yolo | nafnet\n"
               << "  --no-json             Skip JSON export\n"
               << "  --no-csv              Skip CSV export\n"
               << "  --no-gpu-stats        Skip GPU resource sampling\n"
               << "  --help                Show this help\n";
 }
 
 // Parse CLI args and apply overrides onto a BenchmarkConfig loaded from YAML.
 static BenchmarkConfig load_config(int argc, char* argv[]) {
     // Collect CLI overrides first before parsing
     struct Overrides {
         std::string bench_config_path;
         std::string image_path;
         std::string mode;
         std::string model_type;
         size_t warmup_count = 0;
         double test_duration_sec = -1.0;
         size_t min_iterations = 0;
         int concurrent_clients = 0;
         std::string output_dir;
         bool export_json = false;
         bool export_csv = false;
         bool show_gpu_stats = false;
         bool has_warmup = false;
         bool has_duration = false;
         bool has_min_iterations = false;
         bool has_clients = false;
     } ov;
 
     for (int i = 1; i < argc; ++i) {
         std::string arg = argv[i];
 
         auto get_next = [&]() -> std::string {
             if (i + 1 < argc) return argv[++i];
             throw std::runtime_error(std::string("Missing value for ") + arg);
         };
 
         if (arg == "--config")         ov.bench_config_path = get_next();
         else if (arg == "--image")     ov.image_path = get_next();
         else if (arg == "--warmup")    { ov.warmup_count = std::stoull(get_next()); ov.has_warmup = true; }
         else if (arg == "--duration")  { ov.test_duration_sec = std::stod(get_next()); ov.has_duration = true; }
         else if (arg == "--min-iterations") { ov.min_iterations = std::stoull(get_next()); ov.has_min_iterations = true; }
         else if (arg == "--clients")   { ov.concurrent_clients = std::stoi(get_next()); ov.has_clients = true; }
         else if (arg == "--mode")      ov.mode = get_next();
         else if (arg == "--output-dir") ov.output_dir = get_next();
         else if (arg == "--model")     ov.model_type = get_next();
         else if (arg == "--no-json")   ov.export_json = true;
         else if (arg == "--no-csv")    ov.export_csv = true;
         else if (arg == "--no-gpu-stats") ov.show_gpu_stats = true;
         else if (arg == "--help")      { print_usage(argv[0]); std::exit(0); }
         else throw std::runtime_error("Unknown argument: " + arg);
     }
 
     if (ov.bench_config_path.empty()) {
         print_usage(argv[0]);
         throw std::runtime_error("--config is required.");
     }
     if (ov.mode != "sync" && ov.mode != "async" && ov.mode != "batch" && ov.mode != "all")
         throw std::runtime_error("--mode must be one of: sync, async, batch, all");
 
     // Load YAML config as the base
     BenchmarkConfig config = parse_benchmark_config(ov.bench_config_path);
 
     // Apply CLI overrides on top
     if (!ov.image_path.empty())        config.image_path = ov.image_path;
     if (ov.has_warmup)                 config.warmup_count = ov.warmup_count;
     if (ov.has_duration)               config.test_duration_sec = ov.test_duration_sec;
     if (ov.has_min_iterations)         config.min_iterations = ov.min_iterations;
     if (ov.has_clients)                config.concurrent_clients = ov.concurrent_clients;
     if (!ov.output_dir.empty())        config.output_dir = ov.output_dir;
     if (ov.export_json)                config.export_json = false;
     if (ov.export_csv)                 config.export_csv = false;
     if (ov.show_gpu_stats)             config.show_gpu_stats = false;
 
     config.mode = ov.mode.empty() ? config.mode : ov.mode;
     config.model_type = ov.model_type.empty() ? config.model_type : ov.model_type;
 
     return config;
 }
 
 // ============================================================
 // Benchmark execution — dispatch based on model type + mode
 // ============================================================
 
 template <class Runner>
 std::vector<BenchmarkResult> execute_benchmark(
     const YAML::Node& base_config,
     const BenchmarkConfig& config)
 {
     BenchmarkRunner<Runner> bench;
 
 #ifdef ENABLE_CUDA
     if (config.show_gpu_stats) {
         bench.set_gpu_sampler(std::make_unique<GpuResourceSampler>());
     }
 #endif
 
     std::vector<BenchmarkResult> results;
 
     if (config.mode == "sync" || config.mode == "all") {
         std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
         std::cout << "  Running SyncScheduler benchmark...\n";
         std::cout << "  (batch="
                   << (config.sync_override.batch > 0 ? std::to_string(config.sync_override.batch) : "base")
                   << ")\n";
         std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
         results.push_back(bench.run_sync(base_config, config));
     }
 
     if (config.mode == "async" || config.mode == "all") {
         std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
         std::cout << "  Running AsyncScheduler benchmark...\n";
         std::cout << "  (batch="
                   << (config.async_override.batch > 0 ? std::to_string(config.async_override.batch) : "base")
                   << ")\n";
         std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
         results.push_back(bench.run_async(base_config, config));
     }
 
 #ifdef ENABLE_CUDA
     if (config.mode == "batch" || config.mode == "all") {
         std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
         std::cout << "  Running BatchScheduler benchmark...\n";
         std::cout << "  (batch="
                   << (config.batch_override.batch > 0 ? std::to_string(config.batch_override.batch) : "base")
                   << ")\n";
         std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
         results.push_back(bench.run_batch(base_config, config));
     }
 #else
     if (config.mode == "batch") {
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
         // Load config from YAML, apply CLI overrides
         BenchmarkConfig config = load_config(argc, argv);
 
         // Init logger — suppress noisy logs during benchmark
         logger::Init(logger::LOGLEVEL_WARN);
 
         std::cout << "╔══════════════════════════════════════════════════════════╗\n";
         std::cout << "║         ONNXRuntime Inference Benchmark Tool             ║\n";
         std::cout << "╚══════════════════════════════════════════════════════════╝\n";
 
         std::cout << "\nConfiguration:\n";
         std::cout << "  Model config:     " << config.config_path << "\n";
         std::cout << "  Image:            " << config.image_path << "\n";
         std::cout << "  Model type:       " << config.model_type << "\n";
         std::cout << "  Mode:             " << config.mode << "\n";
         std::cout << "  Warmup:           " << config.warmup_count << " iterations\n";
         std::cout << "  Duration:         " << config.test_duration_sec << " seconds\n";
         std::cout << "  Min iterations:   " << config.min_iterations << "\n";
         std::cout << "  Concurrent:       " << config.concurrent_clients << " client(s)\n";
         std::cout << "  Sync batch:       "
                   << (config.sync_override.batch > 0 ? std::to_string(config.sync_override.batch) : "base")
                   << "  |  Async batch: "
                   << (config.async_override.batch > 0 ? std::to_string(config.async_override.batch) : "base")
                   << "  |  Batch batch: "
                   << (config.batch_override.batch > 0 ? std::to_string(config.batch_override.batch) : "base")
                   << "\n";
 
         // Verify inputs
         if (!fs::exists(config.image_path)) {
             std::cerr << "Error: Image not found: " << config.image_path << "\n";
             return 1;
         }
         if (!fs::exists(config.config_path)) {
             std::cerr << "Error: Model config not found: " << config.config_path << "\n";
             return 1;
         }
 
         // Load base model config YAML — each scheduler mode will apply
         // its own override on top of this base to create an independent Runner.
         YAML::Node base_config = YAML::LoadFile(config.config_path);
 
         std::vector<BenchmarkResult> results;
 
         if (config.model_type == "yolo") {
             results = execute_benchmark<YoloDetector>(base_config, config);
         }
         else if (config.model_type == "nafnet") {
             results = execute_benchmark<NAFNet>(base_config, config);
         }
         else {
             std::cerr << "Unknown model type: " << config.model_type
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
 