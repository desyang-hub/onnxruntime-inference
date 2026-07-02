/**
 * @FilePath     : /onnxruntime-inference/examples/yolov8_infer_demo/yolov8_infer_demo.cpp
 * @Description  :  
 * @Author       : desyang
 * @Date         : 2026-06-30 21:23:30
 * @LastEditors  : desyang
 * @LastEditTime : 2026-07-02 20:30:56
**/
#include "yolov8_infer.h"

void test() {
    const std::string& model_path = "models/yolov8n.onnx";
    Yolov8Infer yolov8_infer(model_path);

    std::string img_path = "assets/bus.png";
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
