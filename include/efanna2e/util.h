//
// Created by 付聪 on 2017/6/21.
//

#ifndef EFANNA2E_UTIL_H
#define EFANNA2E_UTIL_H

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>


#ifdef __APPLE__
#else
    #include <malloc.h>
#endif

#ifdef __AVX__
    #define DATA_ALIGN_FACTOR 8
#else
    #ifdef __SSE2__
        #define DATA_ALIGN_FACTOR 4
    #else
        #define DATA_ALIGN_FACTOR 1
    #endif
#endif

namespace efanna2e {

static void GenRandom(std::mt19937& rng, unsigned* addr, unsigned size, unsigned N) {
    for (unsigned i = 0; i < size; ++i) {
        addr[i] = rng() % (N - size);
    }
    std::sort(addr, addr + size);
    for (unsigned i = 1; i < size; ++i) {
        if (addr[i] <= addr[i - 1]) {
            addr[i] = addr[i - 1] + 1;
        }
    }
    unsigned off = rng() % N;
    for (unsigned i = 0; i < size; ++i) {
        addr[i] = (addr[i] + off) % N;
    }
}

inline float* data_align(float* data_ori, unsigned point_num, unsigned& dim) {
    float* data_new = 0;
    unsigned new_dim = (dim + DATA_ALIGN_FACTOR - 1) / DATA_ALIGN_FACTOR * DATA_ALIGN_FACTOR;
    // std::cout << "align to new dim: "<<new_dim << std::endl;
    struct alignas(DATA_ALIGN_FACTOR * 4) OverAligned {
        char b;
    };
    size_t num_bytes = point_num * new_dim * sizeof(float);
    size_t num_aligned = (num_bytes + sizeof(OverAligned) - 1) / sizeof(OverAligned);
    data_new = (float*)new OverAligned[num_aligned];

    for (unsigned i = 0; i < point_num; i++) {
        memcpy(data_new + i * new_dim, data_ori + i * dim, dim * sizeof(float));
        memset(data_new + i * new_dim + dim, 0, (new_dim - dim) * sizeof(float));
    }
    dim = new_dim;
    delete[] data_ori;
    return data_new;
}
}  // namespace efanna2e

/**
 * Returns the peak (maximum so far) resident set size (physical
 * memory use) measured in bytes, or zero if the value cannot be
 * determined on this OS.
 */
size_t getPeakRSS();

/**
 * Returns the current resident set size (physical memory use) measured
 * in bytes, or zero if the value cannot be determined on this OS.
 */
size_t getCurrentRSS();

#endif  // EFANNA2E_UTIL_H
