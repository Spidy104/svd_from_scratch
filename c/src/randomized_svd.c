#include "randomized_svd.h"

#include <float.h>
#include <math.h>
#include <string.h>

static size_t width_of(const randomized_svd_c64 *factor) {
    if (factor == nullptr) return 0;
    const size_t modes = factor->rows < factor->cols ? factor->rows : factor->cols;
    if (factor->rank == 0 || factor->rank > modes) return 0;
    const size_t extra = factor->oversampling < modes - factor->rank ? factor->oversampling : modes - factor->rank;
    return factor->rank + extra;
}

size_t randomized_svd_c64_work_size(const randomized_svd_c64 *factor) {
    const size_t width = width_of(factor);
    if (width == 0 || factor->cols > (SIZE_MAX - factor->rows) / 4) return 0;
    const size_t base = factor->rows + 4 * factor->cols;
    if (width > SIZE_MAX - base || width > SIZE_MAX / (base + width)) return 0;
    const size_t elements = width * (base + width);
    return elements <= SIZE_MAX / sizeof(double complex) ? elements : 0;
}

size_t randomized_svd_c64_real_work_size(const randomized_svd_c64 *factor) {
    const size_t width = width_of(factor);
    return width <= SIZE_MAX / (2 * sizeof(double)) ? 2 * width : 0;
}

static double uniform(uint64_t *state) {
    *state += UINT64_C(0x9e3779b97f4a7c15);
    uint64_t value = *state;
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    value ^= value >> 31;
    return ((double)(value >> 11) + 0.5) * 0x1p-53;
}

static double norm_of(const double complex *column, size_t rows) {
    double total = 0.0;
    for (size_t i = 0; i < rows; ++i) total += creal(column[i]) * creal(column[i]) + cimag(column[i]) * cimag(column[i]);
    return sqrt(total);
}

static void project(double complex *q, size_t rows, size_t column) {
    double complex *current = q + column * rows;
    for (size_t pass = 0; pass < 2; ++pass) for (size_t j = 0; j < column; ++j) {
        const double complex *previous = q + j * rows;
        double complex coefficient = 0.0;
        for (size_t i = 0; i < rows; ++i) coefficient += conj(previous[i]) * current[i];
        for (size_t i = 0; i < rows; ++i) current[i] -= coefficient * previous[i];
    }
}

static svd_status orthonormalize(double complex *q, size_t rows, size_t columns) {
    for (size_t j = 0; j < columns; ++j) {
        double complex *column = q + j * rows;
        const double original = norm_of(column, rows);
        project(q, rows, j);
        double norm = norm_of(column, rows);
        if (norm <= 64.0 * DBL_EPSILON * original) {
            norm = 0.0;
            for (size_t candidate = 0; candidate < rows; ++candidate) {
                memset(column, 0, rows * sizeof(*column));
                column[candidate] = 1.0;
                project(q, rows, j);
                norm = norm_of(column, rows);
                if (norm > sqrt(DBL_EPSILON)) break;
            }
            if (norm <= sqrt(DBL_EPSILON)) return SVD_NO_CONVERGENCE;
        }
        for (size_t i = 0; i < rows; ++i) column[i] /= norm;
    }
    return SVD_OK;
}

static void multiply(double complex *output, const double complex *a, const double complex *b, size_t rows, size_t inner, size_t columns, double scale) {
    memset(output, 0, rows * columns * sizeof(*output));
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(rows * inner * columns >= 1048576)
#endif
    for (size_t j = 0; j < columns; ++j) for (size_t k = 0; k < inner; ++k) {
        const double complex coefficient = b[k + j * inner];
        for (size_t i = 0; i < rows; ++i) output[i + j * rows] += a[i + k * rows] * coefficient;
    }
    if (scale != 1.0) for (size_t i = 0; i < rows * columns; ++i) output[i] /= scale;
}

static void adjoint_multiply(double complex *output, const double complex *a, const double complex *b, size_t rows, size_t inner, size_t columns, double scale) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(rows * inner * columns >= 1048576)
#endif
    for (size_t j = 0; j < columns; ++j) for (size_t k = 0; k < inner; ++k) {
        double complex value = 0.0;
        for (size_t i = 0; i < rows; ++i) value += conj(a[i + k * rows]) * b[i + j * rows];
        output[k + j * inner] = value / scale;
    }
}

svd_status randomized_svd_c64_run(randomized_svd_c64 *factor, const double complex *matrix) {
    if (randomized_svd_c64_work_size(factor) == 0 || matrix == nullptr || factor->u == nullptr || factor->v == nullptr || factor->s == nullptr || factor->work == nullptr || factor->real_work == nullptr) return SVD_BAD_ARGUMENT;
    const size_t rows = factor->rows;
    const size_t cols = factor->cols;
    const size_t width = width_of(factor);
    double complex *q = factor->work;
    double complex *z = q + rows * width;
    double complex *b = z + cols * width;
    double complex *left = b + width * cols;
    double complex *right = left + width * width;
    double complex *work = right + cols * width;
    double *s = factor->real_work;
    double *e = s + width;
    double scale = 0.0;
    for (size_t i = 0; i < rows * cols; ++i) {
        const double magnitude = cabs(matrix[i]);
        if (!isfinite(magnitude)) return SVD_BAD_ARGUMENT;
        scale = fmax(scale, magnitude);
    }
    scale = scale == 0.0 || (scale >= 1e-100 && scale <= 1e100) ? 1.0 : fmax(scale, DBL_MIN);
    uint64_t state = factor->seed;
    for (size_t i = 0; i < cols * width; ++i) {
        const double radius = sqrt(-log(uniform(&state)));
        const double angle = 6.2831853071795864769 * uniform(&state);
        z[i] = radius * (cos(angle) + I * sin(angle));
    }
    multiply(q, matrix, z, rows, cols, width, scale);
    svd_status status = orthonormalize(q, rows, width);
    if (status != SVD_OK) return status;
    for (size_t iteration = 0; iteration < factor->power_iterations; ++iteration) {
        adjoint_multiply(z, matrix, q, rows, cols, width, scale);
        status = orthonormalize(z, cols, width);
        if (status != SVD_OK) return status;
        multiply(q, matrix, z, rows, cols, width, scale);
        status = orthonormalize(q, rows, width);
        if (status != SVD_OK) return status;
    }
    adjoint_multiply(b, q, matrix, rows, width, cols, scale);
    svd_c64 reduced = {width, cols, left, s, right, work, e};
    status = svd_c64_run(&reduced, b);
    if (status != SVD_OK) return status;
    multiply(factor->u, q, left, rows, width, factor->rank, 1.0);
    memcpy(factor->v, right, cols * factor->rank * sizeof(*right));
    memcpy(factor->s, s, factor->rank * sizeof(*s));
    for (size_t i = 0; i < factor->rank; ++i) factor->s[i] *= scale;
    return SVD_OK;
}
