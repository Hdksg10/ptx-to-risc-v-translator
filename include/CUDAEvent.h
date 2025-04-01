#include <cuda_runtime.h>
#include <chrono>

extern "C" {
    struct CUevent_st {
        size_t record_time = 0;
    };
}

namespace driver {
    
}