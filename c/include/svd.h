#ifndef SVD_H
#define SVD_H

#include <complex.h>
#include <stddef.h>

typedef enum {
    SVD_OK,
    SVD_BAD_ARGUMENT,
    SVD_NO_CONVERGENCE
} svd_status;

typedef struct {
    size_t rows;
    size_t cols;
    double complex *u;
    double *s;
    double complex *v;
    double complex *work;
    double *superdiagonal;
} svd_c64;

[[nodiscard]] svd_status svd_c64_run(svd_c64 *factor, const double complex *matrix);

#endif
