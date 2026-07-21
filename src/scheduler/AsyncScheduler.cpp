#include "scheduler/AsyncScheduler.h"

#include "runner/detect/YoloDetector.h"
#include "runner/image_restoration/NAFNet.h"

template class AsyncScheduler<YoloDetector>;
template class AsyncScheduler<NAFNet>;