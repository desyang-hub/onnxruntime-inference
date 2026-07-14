cmake_build_type ?= Debug
cmake_build_dir   = build
with_cuda ?= ON # OFF # ON

.PHONY: all clean release debug

all: $(cmake_build_dir)/CMakeCache.txt
	cmake --build $(cmake_build_dir) --config $(cmake_build_type) \
	  -j$(shell nproc 2>/dev/null || echo %NUMBER_OF_PROCESSORS%)

$(cmake_build_dir)/CMakeCache.txt: Makefile CMakeLists.txt
	cmake -B $(cmake_build_dir) -DCMAKE_BUILD_TYPE=$(cmake_build_type) -DENABLE_CUDA=${with_cuda}

release:
	$(MAKE) cmake_build_type=Release

debug:
	$(MAKE) cmake_build_type=Debug

clean:
	cmake --build $(cmake_build_dir) --target clean \
	  -j$(shell nproc 2>/dev/null || echo %NUMBER_OF_PROCESSORS%)
# -DOpenCV_DIR="path/to/opencv/lib" # windows添加
# examples:
# cmake -B build -DENABLE_CUDA=OFF -DOpenCV_DIR="E:/local/opencv-4.10.0/opencv/build/x64/vc16/lib"
# cmake --build build --config Release -j4

# -DENABLE_CUDA=ON # 用于启用CUDA