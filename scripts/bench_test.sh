#!/bin/bash

CLIENTS=(1 2 4 8 16 32 64)
OUTPUT_DIR="benchmark_output/results/$(date '+%Y%m%d_%H%M%S')"
BENCHMARK_CONFIG="benchmark/benchmark_config.yaml"
TEST_MODE=all # [sync | async | batch | all]

mkdir -p "$OUTPUT_DIR"

for c in "${CLIENTS[@]}"; do
    echo "=========================================="
    echo "=> 开始测试: clients=$c  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="

    ./bin/benchmark_cli --config "$BENCHMARK_CONFIG" --mode $TEST_MODE --clients "$c" 2>&1 | tee "$OUTPUT_DIR/clients_${c}.log"

    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "⚠️  clients=$c 执行失败 (exit code: $exit_code)"
        # 根据需要选择：继续下一个 / 中断退出
        # break
    fi

    echo ""
done

echo "=> 全部测试完成 $(date '+%Y-%m-%d %H:%M:%S')"