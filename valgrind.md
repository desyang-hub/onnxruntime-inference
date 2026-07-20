Valgrind 是 Linux 下最权威的内存调试工具，特别适合排查你遇到的 `tcache unaligned chunk` 这类堆损坏问题。以下是完整的安装与针对你项目的实战使用指南。

### 1. 安装 Valgrind

根据你的系统发行版选择对应命令：

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install -y valgrind

# CentOS / RHEL / Rocky Linux
sudo yum install -y valgrind   # 或 dnf install -y valgrind

# Arch Linux
sudo pacman -S valgrind

# 验证安装及版本（建议 3.18+ 以获得更好的 C++/CUDA 支持）
valgrind --version
```

> ⚠️ **注意**：Valgrind 仅支持 Linux x86_64/aarch64，不支持 Windows/macOS。你的环境是 X620-G40 服务器，完全兼容。

### 2. 重新编译你的项目（关键步骤）

Valgrind 不需要特殊编译选项，但为了获得**带行号的精确调用栈**，必须确保有调试符号：

```cmake
# CMakeLists.txt 中确保 mini_batch_infer 包含调试信息
target_compile_options(mini_batch_infer PRIVATE -g -O1)
# ❌ 不要加 -fsanitize=address，Valgrind 与 ASan 不能共存
# ✅ -O1 比 -O0 好：保留足够调试信息的同时避免过度优化导致误报
```

重新编译：
```bash
cd build && cmake .. && make mini_batch_infer -j$(nproc)
```

### 3. 针对你场景的运行命令

#### 🔍 基础堆损坏检测（首选）
```bash
valgrind \
  --tool=memcheck \
  --track-origins=yes \
  --leak-check=full \
  --show-leak-kinds=all \
  --log-file=valgrind_out.log \
  ./bin/mini_batch_infer
```

| 参数 | 作用 |
|------|------|
| `--tool=memcheck` | 内存错误检测（越界、UAF、未初始化读取、double-free） |
| `--track-origins=yes` | **追踪未初始化值的来源**，对定位 tensor buffer 问题极有用 |
| `--leak-check=full` | 完整内存泄漏报告 |
| `--log-file=...` | 输出到文件，避免与你的 spdlog 日志混在一起 |

#### 🧵 多线程竞态检测（如果 memcheck 没发现问题）
你的崩溃发生在 Thread 14 退出时，可能是并发导致的堆损坏：
```bash
valgrind \
  --tool=helgrind \
  --log-file=helgrind_out.log \
  ./bin/mini_batch_infer
```

> 💡 **Helgrind vs DRD**：两者都是线程错误检测器。Helgrind 对锁顺序和 data race 更敏感；DRD 对 OpenMP/线程池模式误报更少。建议先试 Helgrind。

### 4. 如何解读输出

Valgrind 的输出格式如下，重点关注 **"Invalid write/read"** 和 **"Mismatched free"**：

```text
==12345== Invalid write of size 8
==12345==    at 0x4C3A1B2: TensorBufferPool::release(void*) (TensorBufferPool.cpp:87)
==12345==    by 0x4C3B4E1: OrtSessionWrapper::run(...) (OrtSessionWrapper.h:508)
==12345==    by 0x402F3A: worker_thread(int) (mini_batch_infer.cpp:142)
==12345==  Address 0x7fff37840000 is 0 bytes inside a block of size 1024 free'd
==12345==    at 0x48397AF: cudaFree (vg_replace_malloc.c:...)
==12345==    by 0x4C3A0F8: TensorBufferPool::~TensorBufferPool() (TensorBufferPool.cpp:45)
==12345==  Block was alloc'd at
==12345==    at 0x483877F: cudaMalloc (vg_replace_malloc.c:...)
==12345==    by 0x4C39FE0: TensorBufferPool::allocate(unsigned long) (TensorBufferPool.cpp:62)
```

这段输出直接告诉你：
-   **谁写了非法地址**：`TensorBufferPool::release()` 第 87 行
-   **这块内存什么时候被释放的**：析构函数中 `cudaFree`
-   **这块内存什么时候分配的**：`allocate()` 第 62 行

### 5. ⚠️ CUDA + Valgrind 注意事项

Valgrind 对 CUDA 的支持有限，需要注意以下几点：

1.  **CUDA API 包装**：Valgrind 3.18+ 内置了对 `cudaMalloc/cudaFree` 的基本跟踪，但对 kernel launch 内部的显存访问**无法检测**。如果你的 bug 在 GPU kernel 内部，请用 `compute-sanitizer`。
2.  **性能下降**：memcheck 慢 20-50x，helgrind 慢 100-300x。测试时用**小 batch、少量图片**，不要跑完整数据集。
3.  **抑制 ORT 内部误报**：ONNX Runtime 内部有一些已知的无害内存模式会被 Valgrind 误报。创建 `valgrind_suppr.txt`：
    ```text
    {
       ort_internal_false_positive
       Memcheck:Cond
       obj:*/libonnxruntime.so*
    }
    {
       ort_cuda_provider_fp
       Memcheck:Value8
       obj:*/libonnxruntime_providers_cuda.so*
    }
    ```
    运行时加 `--suppressions=valgrind_suppr.txt`。
4.  **如果 Valgrind 直接崩溃或挂起**：说明 ORT 使用了 Valgrind 不支持的指令（如某些 AVX-512）。此时回退到 `MALLOC_CHECK_=3` + GDB watchpoint 方案：
    ```bash
    export MALLOC_CHECK_=3
    gdb ./bin/mini_batch_infer
    (gdb) catch syscall munmap
    (gdb) run
    ```

### 📋 推荐调试流程

```
Step 1: valgrind --tool=memcheck --track-origins=yes → 找到确切的非法写入点
         ↓ 如果找到了 → 修复 TensorBuffer Pool 的实现
         ↓ 如果没有 / Valgrind 跑不起来
Step 2: MALLOC_CHECK_=3 + GDB → 提前捕获堆损坏
         ↓
Step 3: compute-sanitizer --tool memcheck → 检查 GPU 侧内存错误
         ↓
Step 4: valgrind --tool=helgrind → 检查多线程竞态
```

先从 Step 1 开始，大概率能直接定位到你的 TensorBuffer Pool 中具体哪一行写坏了堆元数据。