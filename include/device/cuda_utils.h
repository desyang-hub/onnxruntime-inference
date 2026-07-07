/* 用于定义一些不同设备下可能冲突的代码或头文件声明 */
#pragma once

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif