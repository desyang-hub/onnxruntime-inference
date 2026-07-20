#pragma once

#include "runner/detect/Detector.h"
#include "runner/image_restoration/Restorer.h"

#include "runner/detect/YoloDetector.h"
#include "runner/image_restoration/NAFNet.h"

template<class T, class Enable = void>
struct runner_type_trait;

template<class T>
struct runner_type_trait<T, std::enable_if_t<std::is_base_of_v<Detector, T>>> {
    using InputType     = typename T::InputType;
    using OutputType    = typename T::OutputType;
};

template<class T>
struct runner_type_trait<T, std::enable_if_t<std::is_base_of_v<Restorer, T>>> {
    using InputType     = typename T::InputType;
    using OutputType    = typename T::OutputType;
};

// template<>
// struct runner_type_trait<YoloDetector> {
//     using InputType     = YoloDetector::InputType;
//     using OutputType    = YoloDetector::OutputType;
// };

// template<>
// struct runner_type_trait<NAFNet> {
//     using InputType     = NAFNet::InputType;
//     using OutputType    = NAFNet::OutputType;
// };