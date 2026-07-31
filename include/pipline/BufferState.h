#pragma once

enum class BufferState {
    IDLE,           // 空闲，可以使用
    PREPROCESSING,  // CPU正在预处理
    READY,          // 数据已准备好
    H2D_IN_PROGRESS,// GPU正在H2D拷贝
    COMPUTING,      // GPU正在推理
    D2H_IN_PROGRESS,// GPU正在D2H拷贝
    POSTPROCESSING, // CPU正在后处理
    COMPLETED       // 完成，等待释放
};