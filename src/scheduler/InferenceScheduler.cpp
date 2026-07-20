#include "scheduler/InferenceScheduler.h"
#include "runner/detect/YoloDetector.h"
#include "runner/image_restoration/NAFNet.h"

template class InferenceScheduler<YoloDetector>;
template class InferenceScheduler<NAFNet>;