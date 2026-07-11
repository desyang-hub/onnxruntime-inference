#include "scheduler/stage/Stage.h"

#include "runner/detect/Detector.h"
#include "runner/image_restoration/Restorer.h"

template class Stage<cv::Mat, TensorBuffer>;
template class Stage<TensorBuffer, ModelOutput>;
template class Stage<ModelOutput, std::vector<std::vector<Detection>>>;
template class Stage<ModelOutput, std::vector<cv::Mat>>;

template class PreStage<Detector>;
template class PreStage<Restorer>;

template class InferStage<Detector>;
template class InferStage<Restorer>;

template class PostStage<Detector>;
template class PostStage<Restorer>;