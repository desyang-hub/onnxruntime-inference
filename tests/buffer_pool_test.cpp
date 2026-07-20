#include "BufferPool.h"
#include "logger/logger.h"

int main(int argc, char const *argv[])
{
    
    BufferPool pool_(4, sizeof(float));

    for (int i = 0; i < 8; ++i)
        LOG_INFO("data ptr: {}", fmt::ptr(pool_.Acquire<float>().get()));

    return 0;
}
