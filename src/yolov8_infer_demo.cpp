/**
 * @FilePath     : /onnxruntime-infer/src/yolov8_infer_demo.cpp
 * @Description  :  
 * @Author       : desyang
 * @Date         : 2026-06-30 21:23:30
 * @LastEditors  : desyang
 * @LastEditTime : 2026-06-30 21:35:46
**/
#include "yolov8_infer.h"
#include "ModelConfig.h"


void test() {
    ModelConfig config{"/home/desyang/github/onnxruntime-infer/models/yolov8n.onnx", true};
    Yolov8Infer yolov8_infer(config);

    std::string img_path = "/home/desyang/github/onnxruntime-infer/assets/bus.png";
    cv::Mat img = cv::imread(img_path);

    yolov8_infer.run(img);

    std::cout << "推理结束" << std::endl;
}

int main(int argc, char const *argv[])
{
    try
    {
        test();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    
    return 0;
}
