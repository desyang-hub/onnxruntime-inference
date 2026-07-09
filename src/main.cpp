#include <iostream>
#include <string>
#include <fstream>
#include <cstddef>
#include <vector>
#include <chrono>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
// #include <cuda_runtime.h>
// #include <cuda_device_runtime_api.h>

#include <filesystem>
namespace fs = std::filesystem;

#ifdef _WIN32
    #include <windows.h>
    // UTF-8 string → wstring
    inline std::wstring utf8_to_wide(const std::string& utf8) {
        if (utf8.empty()) return {};
        int len = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), 
                                    static_cast<int>(utf8.size()), nullptr, 0);
        std::wstring result(len, L'\0');
        MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), 
                            static_cast<int>(utf8.size()), &result[0], len);
        return result;
    }
#endif


int main(int argc, char const *argv[])
{
    std::cout << "Main Run" << std::endl;
    std::string model_path = "models/yolov8n.onnx";
    std::string img_path = "assets/bus.png";

    // 判断路径是否存在
    if (!std::fstream(model_path.c_str()).good()) {
        std::cout << "model not found: " << model_path << std::endl;
        return 0;
    }

    // 初始化ONNXRuntime环境
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "yolov8n_inference");

    // 设置会话选项
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    session_options.SetInterOpNumThreads(1); // 设置线程数量，控制优化可使用的线程数量

    // 获取可用推理设备
    auto providers = Ort::GetAvailableProviders();
    std::cout << "可用推理设备:" << std::endl;
    for (const auto& provider : providers) {
        std::cout << provider << std::endl;
    }

    bool isCudaAvailable = std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") != providers.end();

    if (isCudaAvailable) {
        try
        {
            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = 0;
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic; // 快速启动
            cuda_options.gpu_mem_limit = 2ULL * 1024 * 1024 * 1024;               // 限制 4GB
            cuda_options.do_copy_in_default_stream = 1;

            session_options.AppendExecutionProvider_CUDA(cuda_options);
        }
        catch(const std::exception& e)
        {
            std::cerr << "CUDA device not available: " << e.what() << std::endl;
        }
    } else {
        std::cout << "使用cpu推理" << std::endl;
    }

#ifdef _WIN32
    // 创建推理会话
    Ort::Session session(env, utf8_to_wide(model_path).c_str(), session_options);
#else
    // 创建推理会话
    Ort::Session session(env, model_path.c_str(), session_options);
#endif


    // 输入输出节点的个数
    size_t input_nodes_num = session.GetInputCount();
    size_t output_nodes_num = session.GetOutputCount();

    std::vector<std::string> input_node_names;
    input_node_names.reserve(input_nodes_num);
    std::vector<std::string> output_node_names;
    output_node_names.reserve(output_nodes_num);

    // Ort 默认的内存分配器
    Ort::AllocatorWithDefaultOptions allocator;

    auto shape = 
    session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    int batch = shape[0];
    int channels = shape[1];
    int input_h = shape[2];
    int input_w = shape[3];

    
    for (int i = 0; i < input_nodes_num; ++i) {
        input_node_names.emplace_back(session.GetInputNameAllocated(i, allocator).get());
    }

    // 获取输出的节点信息
    auto outputInfo = 
    session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    int numBatch        = outputInfo[0];
    int numAttributes   = outputInfo[1];
    int numPredictions  = outputInfo[2];
    
    std::cout << "Output shape: [" << numBatch << ", " << numPredictions << ", " << numAttributes << "]" << std::endl;

    for (int i = 0; i < output_nodes_num; ++i) {
        output_node_names.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
    }

    
    cv::Mat img = cv::imread(img_path);
    int img_width = img.cols;
    int img_height = img.rows;

    // 将原始图像填充为正方形，多余部分补充零
    int maxLen = img_width > img_height ? img_width : img_height;
    cv::Mat paddingImg = cv::Mat::zeros(cv::Size(maxLen, maxLen), CV_8UC3);
    cv::Rect roi(0, 0, img_width, img_height);
    img.copyTo(paddingImg(roi)); // 将图像粘贴到左上角

    // 缩放因子，用于将输出坐标缩放会padding图像
    float x_factor = maxLen / static_cast<float>(input_w);
    float y_factor = maxLen / static_cast<float>(input_h);

    // blobFromImage 将图像转换为4 dim张量
    cv::Mat blobTensor = 
    cv::dnn::blobFromImage(paddingImg, 1.0 / 255.0, cv::Size(input_w, input_h), cv::Scalar(), true, false);

    // blobTensor中的像素点个数 b * c * w * h
    size_t pixel_count = channels * input_h * input_w;
    // 定义输入张量的形状
    std::vector<int64_t> input_shape_info{1, channels, input_h, input_w};


    // 准备数据输入
    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU); // 指定张量存在CPU中

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        allocator_info, 
        blobTensor.ptr<float>(), 
        blobTensor.total(), 
        input_shape_info.data(), 
        input_shape_info.size()
    );

    std::vector<const char*> inputNames{input_node_names[0].c_str()};
    std::vector<const char*> outputNames{output_node_names[0].c_str()};


    // 推理开始计时
    auto infer_start_time = std::chrono::high_resolution_clock::now();

    // 模型预热
    size_t dummy_input_size = batch * channels * input_h * input_w;
    std::vector<float> dummy_input_data(dummy_input_size, 0.0f);

    std::vector<int64_t> dummy_input_shape{batch, channels, input_h, input_w};
    auto dummy_allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value dummy_input_tensor = Ort::Value::CreateTensor<float>(
        dummy_allocator_info,
        dummy_input_data.data(),
        dummy_input_data.size(),
        dummy_input_shape.data(),
        dummy_input_shape.size()
    );

    // 模型预热
    std::cout << "模型正在预热" << std::endl;

    for (int i = 0; i < 3; ++i) {
        try
        {
            session.Run(
                Ort::RunOptions{},
                inputNames.data(),
                &dummy_input_tensor,
                1,
                outputNames.data(),
                outputNames.size()
            );
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            return -1;
        }
    }
    std::cout << "预热完成" << std::endl;

    
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<Ort::Value> ort_outputs;
    try
    {
        ort_outputs = session.Run(
            Ort::RunOptions{},
            inputNames.data(),
            &input_tensor,
            1,
            outputNames.data(),
            outputNames.size()
        );
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return -1;
    }

    auto time_end = std::chrono::high_resolution_clock::now();
    std::cout << "spend times: " <<  std::chrono::duration_cast<std::chrono::milliseconds>(time_end - start_time).count() << " ms";

    std::cout << "推理完成" << std::endl;

    auto inference_end = std::chrono::high_resolution_clock::now();

    // 解析推理结果
    const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
    cv::Mat det_output0(numAttributes, numPredictions, CV_32F, (float*)pdata);

    // 转置，方便后续处理
    cv::Mat det_output;
    cv::transpose(det_output0, det_output);

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;

    	// 遍历所有检测到的候选框 (det_output的每一行代表一个候选框)
	for (int i = 0; i < det_output.rows; i++) {
		//float confidence = det_output.at<float>(i, 4);
		//if (confidence < 0.45) {
		//	continue;
		//}
		
		// 获得当前目标框的所有类别得分
		cv::Mat classes_scores = det_output.row(i).colRange(4, numAttributes);

		cv::Point classIdPoint;		// 用于存储分类中的得分最大值索引(坐标)
		double score;				// 用于存储分类中的得分最大值
		minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);
		// 处理分类得分较高的目标框
		if (score > 0.25)
		{
			// 计算在原始图像上,目标框的中心点坐标和宽高
			// 在输入图像上目标框的中心点坐标和宽高
			float cx = det_output.at<float>(i, 0);
			float cy = det_output.at<float>(i, 1);
			float ow = det_output.at<float>(i, 2);
			float oh = det_output.at<float>(i, 3);
			//原始图像上目标框的左上角坐标
			int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
			int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
			//原始图像上目标框的宽高
			int width = static_cast<int>(ow * x_factor);
			int height = static_cast<int>(oh * y_factor);

			// 记录目标框信息
			cv::Rect box;
			box.x = x;
			box.y = y;
			box.width = width;
			box.height = height;

			boxes.push_back(box);
			classIds.push_back(classIdPoint.x);
			confidences.push_back(score);
		}
	}

	// NMS:非极大值抑制，去除同一目标的多余结果。
	std::vector<int> indexes;	// 存储经过 NMS 后保留的目标框在 boxes 向量中的索引
	if (!boxes.empty()) {
		cv::dnn::NMSBoxes(boxes, confidences, 0.25f, 0.45f, indexes);
	}
	std::cout << "get " << indexes.size() << " boxes after NMS." << std::endl;

	std::vector<std::string> labels = {
        "person",        "bicycle",      "car",           "motorcycle",
        "airplane",      "bus",          "train",         "truck",
        "boat",          "traffic light", "fire hydrant",  "stop sign",
        "parking meter", "bench",        "bird",          "cat",
        "dog",           "horse",        "sheep",         "cow",
        "elephant",      "bear",         "zebra",         "giraffe",
        "backpack",      "umbrella",     "handbag",       "tie",
        "suitcase",      "frisbee",      "skis",          "snowboard",
        "sports ball",   "kite",         "baseball bat",  "baseball glove",
        "skateboard",    "surfboard",    "tennis racket",   "bottle",
        "wine glass",    "cup",          "fork",          "knife",
        "spoon",         "bowl",         "banana",        "apple",
        "sandwich",      "orange",       "broccoli",      "carrot",
        "hot dog",       "pizza",        "donut",         "cake",
        "chair",         "couch",        "potted plant",  "bed",
        "dining table",  "toilet",       "tv",            "laptop",
        "mouse",         "remote",       "keyboard",      "cell phone",
        "microwave",     "oven",         "toaster",       "sink",
        "refrigerator",  "book",         "clock",         "vase",
        "scissors",      "teddy bear",   "hair drier",    "toothbrush"
    };

	// 遍历筛选出的目标框
	for (size_t i = 0; i < indexes.size(); i++) {
        try
        {
            int idx = indexes[i];		// 获取当前目标框序号
            int cid = classIds[idx];	// 获取目标框分类得分

            cv::rectangle(img, boxes[idx], cv::Scalar(0, 0, 255), 1, 8, 0);
            cv::putText(img, labels[cid].c_str(), boxes[idx].br(), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            return -1;
        }
	}

	fs::create_directory("output");
	cv::imwrite("./output/bus.jpg", img);

    
	auto inference_duration = std::chrono::duration_cast<std::chrono::milliseconds>(inference_end - infer_start_time).count();
	std::cout << model_path << "模型推理耗时: " << inference_duration << " ms" << std::endl;

    // cv::imshow("DetectResult", img);
	// cv::waitKey(0);
    

    return 0;
}
