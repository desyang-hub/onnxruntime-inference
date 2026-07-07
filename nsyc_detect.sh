# 精确到微秒级的 GPU 活动分析
nsys profile --trace=cuda,nvtx --stats=true -o yolo_bs4 ./bat

# 查看报告中 "CUDA Kernel Summary" 和 "GPU Occupancy"