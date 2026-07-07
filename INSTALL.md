# 依赖安装
``` bash
# 安装opencv开发环境
sudo apt install libopencv-dev
```

# 安装tensorRT
``` bash
pip install tensorrt==10.*
# sudo apt update
# sudo apt-get install -y libnvinfer10 libnvinfer-plugin10

# 查找 libnvinfer.so.10 的实际位置
find $(python -c "import tensorrt; print(tensorrt.__path__[0])") -name "libnvinfer.so.10"

# 将输出的目录添加到环境变量（替换为实际路径）
export LD_LIBRARY_PATH=/home/team/miniconda3/envs/sealRecPaddle/lib/python3.9/site-packages/tensorrt_libs

# 验证链接是否解析成功
ldd /home/team/yds_workspace/onnxruntime-inference/build/_deps/onnxruntime_prebuilt-src/lib/libonnxruntime_providers_tensorrt.so | grep nvinfer
```