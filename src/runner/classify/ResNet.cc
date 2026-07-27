#include "runner/classify/ResNet.h"

#include "logger/logger.h"

ResNet::ResNet(const YAML::Node& config) : Classifier(config["model"]) {
}

cv::Mat ResNet::resize_256(const cv::Mat& img) const {
    int h = img.rows;
    int w = img.cols;
    int low_len = std::min(h, w);

    const int kTargetLen = 256;

    float scale = round(static_cast<float>(kTargetLen) / low_len);

    int new_h = scale * h;
    int new_w = scale * w;

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));

    return resized;
}

cv::Mat ResNet::center_crop(const cv::Mat& img, const cv::Size& size) const {
    cv::Mat cent = cv::Mat::zeros(size, CV_8UC3);

    int h = img.rows;
    int w = img.cols;

    int pad_left = (w - size.width) / 2;
    int pad_top = (h - size.height) / 2;

    img(cv::Rect(pad_left, pad_top, size.width, size.height)).copyTo(cent);

    return cent;
}

TensorBuffer ResNet::preprocess(const cv::Mat& img) {
    cv::Mat result = resize_256(img);
    result = center_crop(result);

    cv::Mat blob = cv::dnn::blobFromImage(
        result, 1.0 / 255.0, cv::Size(224, 224),
        cv::Scalar(0.485, 0.456, 0.406), true, false, CV_32F);
    
    // blob 是 NCHW [1,3,224,224]，逐通道除以 std
    // OpenCV Mat 对 NCHW 4D Mat 的 Scalar 运算按通道广播
    cv::Scalar std_scalar(0.229, 0.224, 0.225);
    blob /= std_scalar;

    // 将数据移动到GPU
    TensorBuffer tb = backend_->tensorBuffer();

#ifdef ENABLE_CUDA
    TensorBuffer tenbuf = backend_->GetTensorBuffer();
    if (!blob.isContinuous()) {
        throw std::invalid_argument(MESSAGE_WITH_LOC("Mat must be continuous for linear H2D copy"));
    }
    tb.data = tenbuf.data;

    size_t copy_bytes = blob.total() * blob.elemSize();
    cudaMemcpyAsync(tenbuf.data, blob.data, copy_bytes, cudaMemcpyHostToDevice, stream_.get());
    cudaStreamSynchronize(stream_.get());
#endif

    return tb;
}


#ifdef ENABLE_CUDA
void ResNet::preprocess(const cv::Mat&, TensorBuffer&, int offset) {

}
#endif


std::vector<int> ResNet::postprocess(const ModelOutput& model_output) {
    const TensorBuffer& tenbuf = model_output.primary();

    TensorBuffer tb =  TensorBuffer::create(tenbuf.shape);

#ifdef ENABLE_CUDA
    cudaMemcpy(tb.data, tenbuf.data, tenbuf.byte_size(), cudaMemcpyDeviceToHost);
#endif

    int batch = tenbuf.letterbox_params.size();
    int classNum = tenbuf.shape[1];
    const int plane_size = tenbuf.plane_size();
    std::vector<int> res;

    for (int i = 0; i < batch; ++i) {
        float* data = tb.data + i * plane_size;

        int id = std::max_element(data, data + classNum) - data;
        res.push_back(id);
    }

    return res;
}