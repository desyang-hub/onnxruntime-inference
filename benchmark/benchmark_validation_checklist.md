# Benchmark 验证清单

> 在将 Benchmark 工具用于生产级性能报告前，请逐项手动检查以下关键点。

---

## 🔴 P0 — 计时正确性（必须验证）

### 1. GPU 异步操作是否真正完成后再计时结束？
- [ ] **检查点**：`run_sync()` / `run_async()` / `run_batch()` 中，每次 `future.get()` 返回时，确认 GPU 上的推理已完全结束。
- **为什么重要**：如果 `future.get()` 只确认了 postprocess 完成但 GPU kernel 仍在运行，P50/P90 会被低估。
- **如何验证**：在 `fut.get()` 后紧跟 `cudaDeviceSynchronize()` 并对比两次计时。如果差异 > 1ms，说明存在异步泄漏。
- **当前实现**：`BatchScheduler` 中 `infer_stage_` 调用 `backend_->infer()` 后返回 `ModelOutput`，而 `OrtSessionWrapper::infer()` 在 GPU 模式下使用自定义 output buffer 但没有显式同步——**需要确认 ONNXRuntime `session_->Run()` 在 CUDA EP 下是同步还是异步返回**。ONNXRuntime 默认同步等待 kernel 完成，但需在实际硬件上验证。

### 2. Warmup 是否充分？
- [ ] **检查点**：Warmup 后第一次正式测量的延迟是否显著高于稳态？
- **为什么重要**：GPU 频率提升、TensorRT 引擎缓存、CPU cache 预热都需要时间。
- **如何验证**：打印 warmup 后的前 5 次延迟，观察是否呈下降趋势。如果是，增加 `warmup_count`。

### 3. CPU 时钟在跨线程场景下的准确性
- [ ] **检查点**：`std::chrono::high_resolution_clock` 在多核 CPU 上是否单调？
- **为什么重要**：Linux 上 `CLOCK_MONOTONIC` 是安全的，但某些虚拟化环境可能漂移。
- **如何验证**：在 benchmark 开始前打印 `CLOCK_MONOTONIC` vs `CLOCK_REALTIME` 对比。

---

## 🟡 P1 — 资源与公平性（强烈建议验证）

### 4. 三种模式使用相同的输入图像
- [ ] **检查点**：确认 `run_sync()` / `run_async()` / `run_batch()` 使用的是同一张图像的 clone。
- **当前实现**：`main.cpp` 中在 `execute_benchmark()` 外部只读取一次图像，各模式在内部使用 `test_img`。**但 Sync 和 Async 共用同一个 `test_img` cv::Mat 时，如果 preprocess 中有写操作会产生数据竞争**。
- **修复建议**：Batch 模式已用 `test_img.clone()`，Sync/Async 也应 clone 或在 submit 内部 clone。

### 5. 并发客户端数对 Async/Batch 模式的影响
- [ ] **检查点**：`concurrent_clients > 1` 时，AsyncScheduler 的 SyncExecutor 是否会成为瓶颈？
- **为什么重要**：AsyncScheduler 的 infer 阶段是单线程 FIFO，并发提交只会增加队列长度，不会增加 GPU 吞吐。
- **如何验证**：对比 `--clients 1` 和 `--clients 4` 的 Async 模式吞吐，预期变化不大。

### 6. BatchScheduler 的 batch 对齐
- [ ] **检查点**：当 `concurrent_clients` 提交的请求数不能被 `batch_size` 整除时，最后一个不满批次的延迟是否合理？
- **为什么重要**：BatchScheduler 有超时机制（10ms），但超时批次可能拖慢 P99。
- **如何验证**：检查 `latency_p99` 是否异常高。

### 7. GPU 频率锁定
- [ ] **检查点**：测试期间 GPU 频率是否稳定？
- **为什么重要**：NVIDIA GPU 的动态频率调节（boost clock）会导致前几次推理快（boost 未触发）或慢（thermal throttling）。
- **如何验证**：使用 `nvidia-smi -l 1` 监控频率，或使用 `nvidia-smi -pm 1` 锁定性能模式。

---

## 🟢 P2 — 工程细节（推荐验证）

### 8. 内存泄漏
- [ ] **检查点**：长时间运行后显存和内存是否持续增长？
- **如何验证**：使用 `valgrind --leak-check=full`（CPU 模式）或 `nsight systems`（GPU 模式）。

### 9. 日志级别影响性能
- [ ] **检查点**：Benchmark 运行时日志级别设为 `WARN`，确认没有 TRACE/DEBUG 日志污染计时。
- **当前实现**：`main.cpp` 中已设为 `LOGLEVEL_WARN`。

### 10. JSON/CSV 导出格式
- [ ] **检查点**：导出的文件能否被 Excel / Python pandas 正确读取？
- **如何验证**：`python -c "import pandas as pd; df = pd.read_csv('file.csv'); print(df.describe())"`

### 11. _popen 跨平台兼容性
- [ ] **检查点**：`GpuResourceSampler` 使用 `_popen()`，在 Linux 上应使用 `popen()`。
- **当前实现**：⚠️ **使用了 `_popen`（Windows 专有），Linux 编译会报错。需要添加 `#ifdef _WIN32` 条件编译。**

---

## ⚠️ 已知问题与待修复

| # | 问题 | 严重度 | 状态 |
|---|------|--------|------|
| 1 | `_popen` / `_pclose` 是 Windows 专有 API，Linux 应使用 `popen` / `pclose` | 🟡 P1 | 待修复 |
| 2 | `cudaEventElapsedTime` 在 warmup 段调用顺序有误（先赋值后计算） | 🔴 P0 | 待修复 |
| 3 | Sync/Async 模式下共用同一个 `cv::Mat` 可能产生数据竞争 | 🟡 P1 | 待确认 |
| 4 | `benchmark_runner.cpp` 通过 `#include` 方式在 main.cpp 中实例化模板，非标准做法。应考虑改为 header-only 或使用显式实例化 | 🟢 P2 | 可优化 |

---

## 运行验证命令

```bash
# 1. 基础功能验证（单线程，短时长）
./benchmark --config config/model_config.yaml --image assets/bus.png \
            --warmup 3 --duration 5 --clients 1 --mode sync --no-gpu-stats

# 2. 三种模式横向对比
./benchmark --config config/model_config.yaml --image assets/bus.png \
            --warmup 5 --duration 10 --clients 1 --mode all

# 3. 并发压测（多线程提交）
./benchmark --config config/model_config.yaml --image assets/bus.png \
            --warmup 5 --duration 15 --clients 4 --mode async

# 4. Batch 模式专项（需 CUDA）
./benchmark --config config/model_config.yaml --image assets/bus.png \
            --warmup 10 --duration 20 --clients 8 --mode batch

# 5. NAFNet 模型
./benchmark --config config/Restorer.yaml --image assets/seal.png \
            --model nafnet --warmup 5 --duration 10 --mode all
```
