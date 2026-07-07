#include "runner/ModelRunner.h"

#include "backend/OrtSessionWrapper.h"

ModelRunner::ModelRunner(std::unique_ptr<InferenceBackend> backend) : backend_(std::move(backend)) {
    
}

ModelOutput ModelRunner::infer() {
    return backend_->run();
}