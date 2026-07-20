# ONNXRuntime Inference Framework

基于 ONNXRuntime C++ API 的高性能模型推理框架，支持目标检测与图像修复等多类模型的快速接入与高效推理。

## ✨ 特性

- **分层架构**：Backend（硬件抽象）→ Runner（模型适配）→ Scheduler（任务调度）三层解耦，新增模型只需实现 `preprocess()` / `postprocess()` 两个函数
- **多 Execution Provider 支持**：TensorRT → CUDA → CPU 自动降级，通过 YAML 配置驱动，无需修改代码
- **三种推理调度模式**：同步（`SyncScheduler`）、异步流水线（`AsyncScheduler`）、微批推理（`BatchScheduler`）
- **GPU 加速**：自定义 CUDA Kernel 实现端到端预处理（Letterbox + BGR→RGB + 归一化 + HWC→CHW）与后处理过滤
- **零拷贝张量传输**：`TensorBuffer` 基于 `shared_ptr` 实现硬件无关的零拷贝数据传递
- **多级内存池**：GPU 显存池（`InferTensorBufferPool`）+ CPU 内存池（`BufferPool`），避免推理过程中重复分配
- **跨平台**：支持 Windows / Linux，CMake + FetchContent/CPM 自动管理依赖

## 📂 支持的模型

| 模型 | 类型 | 说明 |
|------|------|------|
| YOLOv8 | 目标检测 | Letterbox 预处理、NMS 后处理、支持批量推理 |
| NAFNet | 图像修复 | Padding 预处理、印章去除等图像恢复任务 |

## 🏗 架构

```
┌──────────────────────────────────────────────────────────────┐
│                      Scheduler Layer                         │
│  SyncScheduler │ AsyncScheduler │ BatchScheduler             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  PreStage  →  InferStage  →  PostStage               │   │
│  │  (Concurrent)  (Sync/FIFO)   (Concurrent)             │   │
│  └──────────────────────────────────────────────────────┘   │
├──────────────────────────────────────────────────────────────┤
│                       Runner Layer                           │
│  Detector (YoloDetector)  │  Restorer (NAFNet)               │
│  preprocess() │ infer() │ postprocess()                      │
├──────────────────────────────────────────────────────────────┤
│                      Backend Layer                           │
│  InferenceBackend → OrtSessionWrapper                        │
│  ONNXRuntime Session │ Tensor Cache │ Device Resolution      │
├──────────────────────────────────────────────────────────────┤
│                      Execution Providers                     │
│  TensorRT (FP16/FP32) │ CUDA │ CPU                           │
└──────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 依赖

- **CMake 3.21+**、**C++17** 编译器
- **OpenCV**（`find_package(OpenCV REQUIRED)`）
- **OpenMP**
- （可选）CUDA Toolkit + TensorRT（GPU 推理）

Linux 安装 OpenCV：
```bash
sudo apt install libopencv-dev
```

更多依赖安装说明见 [INSTALL.md](INSTALL.md)。

### 构建

```bash
# CPU 推理
cmake -B build
cmake --build build --config Release

# GPU 推理（CUDA + TensorRT）
cmake -B build -DENABLE_CUDA=ON
cmake --build build --config Release
```

或使用 Makefile：
```bash
make                                    # Debug 构建
make release with_cuda=ON               # Release + CUDA
make clean
```

CMake 选项：
| 选项 | 默认值 | 说明 |
|------|--------|------|
| `ENABLE_CUDA` | `OFF` | 启用 CUDA / TensorRT 推理 |
| `BUILD_EXAMPLES` | `ON` | 构建示例程序 |
| `BUILD_TEST` | `ON` | 构建测试程序 |

构建产物输出到 `bin/`（可执行文件）和 `lib/`（库文件）。

### 运行示例

```bash
cd bin/Release

# YOLO 目标检测
./yolo_infer

# NAFNet 图像修复
./NAFNet_demo
```

运行前需要将 ONNX 模型文件放置在 `models/` 目录，并确认 YAML 配置中的模型路径正确。

## ⚙️ 配置

模型推理参数全部通过 YAML 配置文件管理，无需修改代码。

### YOLO 检测配置 (`config/model_config.yaml`)

```yaml
model:
  path: "models/yolov8n_dynamic.onnx"
  shape: [640, 640]          # 输入尺寸 [H, W]
  batch: 16                  # 批处理大小
  buffer_size: 32            # 缓存区数量
  gpu: 0                     # GPU 设备 ID

  session_options:
    trt_fp16_enable: "1"     # TensorRT FP16 推理
    execution_providers:
      ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
    graph_optimization_level: "ORT_ENABLE_ALL"
    intra_op_num_threads: 4
    inter_op_num_threads: 1
    execution_mode: "ORT_SEQUENTIAL"

preprocess:
  auto_aspect_ratio: true    # Letterbox 等比缩放
  pad_color: [114, 114, 114]
  bgr_to_rgb: true

postprocess:
  conf_threshold: 0.25
  nms_threshold: 0.45
  max_detections: 300
```

### NAFNet 修复配置 (`config/Restorer.yaml`)

```yaml
model:
  path: "models/NAFNet-w32.onnx"
  shape: [576, 576]
  batch: 2
  buffer_size: 1
  gpu: 0
  session_options:
    execution_providers:
      ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
    trt_fp16_enable: "0"
    # ...
```

## 💻 API 使用

### 同步推理

```cpp
#include "runner/detect/YoloDetector.h"

// 加载模型
auto detector = Detector::Load<YoloDetector>("config/model_config.yaml");

// 推理
cv::Mat img = cv::imread("image.jpg");
std::vector<Detection> results = detector->detect(img);

for (const auto& det : results) {
    std::cout << detector->class_label(det.class_id)
              << " " << det.score << std::endl;
}
```

### 异步流水线推理

```cpp
#include "scheduler/AsyncScheduler.h"

auto detector = Detector::Load<YoloDetector>("config/model_config.yaml");
AsyncScheduler<Detector> scheduler(detector);

cv::Mat img = cv::imread("image.jpg");
auto future = scheduler.submit(img);
std::vector<Detection> results = future.get();
```

### 微批推理

```cpp
#ifdef ENABLE_CUDA
#include "scheduler/BatchScheduler.h"

auto runner = Restorer::Load<NAFNet>("config/Restorer.yaml");
auto scheduler = std::make_shared<BatchScheduler<Restorer>>(runner);

cv::Mat img = cv::imread("image.jpg");
auto future = scheduler->submit(img);
cv::Mat result = future.get();
#endif
```

## 📁 项目结构

```
├── include/                    # 头文件
│   ├── backend/               # 硬件抽象层
│   ├── runner/                # 模型适配层
│   │   ├── detect/           # 目标检测
│   │   └── image_restoration/ # 图像修复
│   ├── scheduler/             # 任务调度层
│   │   ├── executor/         # 执行器（并发 / 同步）
│   │   ├── stage/            # 阶段抽象
│   │   └── type_trait/       # 类型特征
│   ├── device/               # CUDA 设备相关
│   └── ...                   # 工具组件
├── src/                       # 实现文件
├── config/                    # YAML 配置文件
├── examples/                  # 示例程序
├── tests/                     # 测试
├── models/                    # ONNX 模型文件
└── cmake/                     # CMake 脚本
```

## 🔧 CUDA Kernel

GPU 模式下，框架使用自定义 CUDA Kernel 加速关键路径：

- **`yolo_preprocess_kernel`**：Letterbox 等比缩放（双线性插值）+ BGR→RGB + 归一化 + HWC→CHW 转置，单次遍历零中间缓冲
- **`filter_and_compact_kernel`**：YOLOv8 后处理过滤，在 GPU 端对 8400 个锚点进行置信度过滤并紧凑化，减少 95%+ 的 GPU→CPU 数据传输

## 📐 数据流

```
输入 (cv::Mat)
    ↓
PreStage (预处理)
    ↓
TensorBuffer (零拷贝张量)
    ↓
InferStage (GPU/CPU 推理)
    ↓
ModelOutput (命名张量集合)
    ↓
PostStage (后处理)
    ↓
输出 (std::vector<Detection> / cv::Mat)
```

## 📄 许可证

[MIT License](LICENSE)
