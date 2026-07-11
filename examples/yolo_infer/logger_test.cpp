#include "logger/logger.h"

int main(int argc, char const *argv[])
{
    // logger::Init(logger::LOGLEVEL_TRACE);
    LOG_TRACE("hhh");
    LOG_DEBUG("hhh");
    LOG_INFO("hhh");
    LOG_WARN("hhh");
    LOG_ERROR("hhh");
    LOG_CRITICAL("hhh");
    
    

    return 0;
}
