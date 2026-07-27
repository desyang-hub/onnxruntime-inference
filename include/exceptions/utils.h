#pragma once

#include <string>
#include <stdexcept>

#define MESSAGE_LOC(msg) \
    (std::string(__FILE__) + ":" + std::to_string(__LINE__))

#define MESSAGE_WITH_LOC(msg) \
    (MESSAGE_LOC(msg) + " " + (msg))

