#ifndef RANDOMIZED_SVD_H
#define RANDOMIZED_SVD_H

#include "svd.h"
#include <stdint.h>

typedef struct {
    size_t rows;
    size_t cols;
    size_t rank;
    size_t oversampling;
    size_t power_iterations;
    uint64_t seed;
    double complex *u;
    double *s;
    double complex *v;
    double complex *work;
    double *real_work;
} randomized_svd_c64;

[[nodiscard]] size_t randomized_svd_c64_work_size(const randomized_svd_c64 *factor);
[[nodiscard]] size_t randomized_svd_c64_real_work_size(const randomized_svd_c64 *factor);
[[nodiscard]] svd_status randomized_svd_c64_run(randomized_svd_c64 *factor, const double complex *matrix);

#endif
