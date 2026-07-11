#include <stdexcept>

#include "runner/ModelRunner.h"
#include "backend/OrtSessionWrapper.h"

int ParseBackendTypeFromString(const std::string& backend) {
    if (backend == "onnxruntime") return BACKEND_ONNXRUNTIME;
    else if (backend == "open_vino") return BACKEND_OPEN_VINO;
    else {
        return -1;
    }
}

ModelRunner::ModelRunner(const YAML::Node& config) {
    int backend_type = ParseBackendTypeFromString(config["backend"].as<std::string>("onnxruntime"));
    switch(backend_type) {
        case BACKEND_ONNXRUNTIME:
            backend_ = std::make_unique<OrtSessionWrapper>(config);
            break;
        default:
            throw std::runtime_error("Model Unsupport");
    }
}

ModelOutput ModelRunner::infer(const TensorBuffer& tb) {
    return backend_->run(tb);
}