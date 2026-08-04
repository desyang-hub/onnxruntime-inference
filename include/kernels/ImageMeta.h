#pragma once

#include <cstdint>

struct ImageMeta {
    int src_h, src_w;
    size_t src_step;
    int right_bt_x, right_bt_y;
    float scale;
    int pad_left, pad_top;
};
