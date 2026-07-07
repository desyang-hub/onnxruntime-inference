all:
	cmake -B build && cmake --build build -j${nproc}
# -DOpenCV_DIR="path/to/opencv/lib" # windows添加
# examples:
# cmake -B build -DOpenCV_DIR="E:/local/opencv-4.10.0/opencv/build/x64/vc16/lib"
# cmake --build build -j4

# -DENABLE_CUDA=ON # 用于启用CUDA