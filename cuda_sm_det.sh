# 每 100ms 采样一次，观察 SM 活跃度和显存带宽利用率
watch -n 0.1 nvidia-smi --query-gpu=utilization.gpu,utilization.memory,power.draw --format=csv
# 或者更精准的 nsight-systems / Nsight Compute