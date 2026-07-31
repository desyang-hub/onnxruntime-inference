#include "queue/ConcurrentQueue.h"
#include "thread/ThreadPool.h"
#include "logger/logger.h"
#include <thread>

int main(int argc, char const *argv[])
{
    ThreadPool pool;
    std::shared_ptr<ConcurrentQueue<int>> que = std::make_shared<ConcurrentQueue<int>>();

    for (int i = 0; i < 10; ++i) {
        pool.enqueue([que, i]{
            int d;
            que->pop(d);
            LOG_INFO("di: {}", d);
        });
    }
    

    for (int i = 0; i < 10; ++i) {
        LOG_INFO("i: {}", i);
        que->push(i);
    }
    

    return 0;
}
