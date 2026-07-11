#pragma once

#include "runner/detect/Detector.h"
#include "runner/image_restoration/Restorer.h"

template<class T>
struct runner_type_trait;


template<>
struct runner_type_trait<Detector> {
    using InputType     = Detector::InputType;
    using OutputType    = Detector::OutputType;
};

template<>
struct runner_type_trait<Restorer> {
    using InputType     = Restorer::InputType;
    using OutputType    = Restorer::OutputType;
};