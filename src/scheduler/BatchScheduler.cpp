#include "scheduler/BatchScheduler.h"
#include "runner/detect/YoloDetector.h"
#include "runner/image_restoration/NAFNet.h"


template class BatchScheduler<YoloDetector>;
template class BatchScheduler<NAFNet>;