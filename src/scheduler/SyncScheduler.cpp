#include "scheduler/SyncScheduler.h"

#include "runner/detect/YoloDetector.h"
#include "runner/image_restoration/NAFNet.h"

template class SyncScheduler<YoloDetector>;
template class SyncScheduler<NAFNet>;