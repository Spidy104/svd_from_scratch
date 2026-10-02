#include "randomized_svd.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {
    struct timespec time;
    timespec_get(&time, TIME_MONOTONIC);
    return time.tv_sec + 1e-9 * time.tv_nsec;
}

static double median(double *values) {
    for (size_t i = 0; i < 3; ++i) for (size_t j = i + 1; j < 3; ++j) if (values[j] < values[i]) {
        const double temporary = values[i];
        values[i] = values[j];
        values[j] = temporary;
    }
    return values[1];
}

static double residual(const double complex *a, const svd_c64 *factor, size_t rank) {
    double squared_error = 0.0;
    double squared_norm = 0.0;
    for (size_t j = 0; j < factor->cols; ++j) for (size_t i = 0; i < factor->rows; ++i) {
        double complex value = 0.0;
        for (size_t k = 0; k < rank; ++k) value += factor->u[i + k * factor->rows] * factor->s[k] * conj(factor->v[j + k * factor->cols]);
        const double complex original = a[i + j * factor->rows];
        const double error = cabs(value - original);
        const double norm = cabs(original);
        squared_error += error * error;
        squared_norm += norm * norm;
    }
    return sqrt(squared_error / fmax(squared_norm, 1e-300));
}

static double orthogonality(const double complex *q, size_t rows, size_t columns) {
    double error = 0.0;
    for (size_t j = 0; j < columns; ++j) for (size_t k = 0; k < columns; ++k) {
        double complex value = j == k ? -1.0 : 0.0;
        for (size_t i = 0; i < rows; ++i) value += conj(q[i + j * rows]) * q[i + k * rows];
        error += creal(value) * creal(value) + cimag(value) * cimag(value);
    }
    return sqrt(error);
}

int main(int argc, char **argv) {
    if (argc != 2) return EXIT_FAILURE;
    FILE *input = fopen(argv[1], "rb");
    if (input == nullptr) return EXIT_FAILURE;
    uint64_t dimensions[3];
    if (fread(dimensions, sizeof(*dimensions), 3, input) != 3) {
        fclose(input);
        return EXIT_FAILURE;
    }
    const size_t rows = dimensions[0];
    const size_t cols = dimensions[1];
    const size_t rank = dimensions[2];
    const size_t modes = rows < cols ? rows : cols;
    if (rows == 0 || cols == 0 || rank == 0 || rank > modes || rows > SIZE_MAX / cols / sizeof(double complex)) {
        fclose(input);
        return EXIT_FAILURE;
    }
    double complex *a = malloc(rows * cols * sizeof(*a));
    double complex *u = malloc(rows * modes * sizeof(*u));
    double complex *v = malloc(cols * modes * sizeof(*v));
    double complex *work = malloc(rows * cols * sizeof(*work));
    double *s = malloc(modes * sizeof(*s));
    double *e = malloc(modes * sizeof(*e));
    randomized_svd_c64 randomized = {rows, cols, rank, 8, 1, 17, u, s, v, nullptr, nullptr};
    randomized.work = malloc(randomized_svd_c64_work_size(&randomized) * sizeof(*randomized.work));
    randomized.real_work = malloc(randomized_svd_c64_real_work_size(&randomized) * sizeof(*randomized.real_work));
    if (a == nullptr || u == nullptr || v == nullptr || work == nullptr || s == nullptr || e == nullptr || randomized.work == nullptr || randomized.real_work == nullptr) return EXIT_FAILURE;
    if (fread(a, sizeof(*a), rows * cols, input) != rows * cols) return EXIT_FAILURE;
    fclose(input);
    svd_c64 exact = {rows, cols, u, s, v, work, e};
    if (svd_c64_run(&exact, a) != SVD_OK || residual(a, &exact, modes) > 1e-10) return EXIT_FAILURE;
    const double optimal = residual(a, &exact, rank);
    double times[3];
    for (size_t run = 0; run < 3; ++run) {
        const double start = now();
        if (svd_c64_run(&exact, a) != SVD_OK) return EXIT_FAILURE;
        times[run] = now() - start;
    }
    const double exact_time = median(times);
    if (randomized_svd_c64_run(&randomized, a) != SVD_OK) return EXIT_FAILURE;
    for (size_t run = 0; run < 3; ++run) {
        const double start = now();
        if (randomized_svd_c64_run(&randomized, a) != SVD_OK) return EXIT_FAILURE;
        times[run] = now() - start;
    }
    const double randomized_time = median(times);
    const double error = residual(a, &exact, rank);
    const double left_error = orthogonality(u, rows, rank);
    const double right_error = orthogonality(v, cols, rank);
    printf("%.9g %.9g %.9g %.9g %.9g %.9g\n", exact_time, randomized_time, optimal, error, left_error, right_error);
    const int valid = isfinite(error) && error <= 1.3 * optimal + 1e-10 && left_error < 1e-10 && right_error < 1e-10;
    free(randomized.real_work); free(randomized.work);
    free(e); free(s); free(work); free(v); free(u); free(a);
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
