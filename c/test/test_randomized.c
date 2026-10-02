#include "randomized_svd.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double orthogonality(const double complex *q, size_t rows, size_t columns) {
    double total = 0.0;
    for (size_t j = 0; j < columns; ++j) for (size_t k = 0; k < columns; ++k) {
        double complex value = j == k ? -1.0 : 0.0;
        for (size_t i = 0; i < rows; ++i) value += conj(q[i + j * rows]) * q[i + k * rows];
        total += creal(value) * creal(value) + cimag(value) * cimag(value);
    }
    return sqrt(total);
}

static int check(size_t rows, size_t cols, size_t rank, size_t iterations, int zero, int real) {
    double complex *a = malloc(rows * cols * sizeof(*a));
    double complex *u = malloc(rows * rank * sizeof(*u));
    double complex *v = malloc(cols * rank * sizeof(*v));
    double complex *saved = malloc(rows * rank * sizeof(*saved));
    double *s = malloc(rank * sizeof(*s));
    if (a == nullptr || u == nullptr || v == nullptr || saved == nullptr || s == nullptr) {
        free(s); free(saved); free(v); free(u); free(a);
        return EXIT_FAILURE;
    }
    for (size_t j = 0; j < cols; ++j) for (size_t i = 0; i < rows; ++i) {
        double complex value = 0.0;
        if (!zero) for (size_t k = 0; k < 3; ++k) {
            const double complex left = sin((i + 1.0) * (k + 0.4)) + (real ? 0.0 : I * cos((i + 0.3) * (k + 1.0)));
            const double complex right = cos((j + 1.0) * (k + 0.7)) + (real ? 0.0 : I * sin((j + 0.2) * (k + 1.0)));
            value += left * right;
        }
        a[i + j * rows] = value;
    }
    randomized_svd_c64 factor = {rows, cols, rank, 8, iterations, 17, u, s, v, nullptr, nullptr};
    factor.work = malloc(randomized_svd_c64_work_size(&factor) * sizeof(*factor.work));
    factor.real_work = malloc(randomized_svd_c64_real_work_size(&factor) * sizeof(*factor.real_work));
    if (factor.work == nullptr || factor.real_work == nullptr) {
        free(factor.real_work); free(factor.work);
        free(s); free(saved); free(v); free(u); free(a);
        return EXIT_FAILURE;
    }
    svd_status status = randomized_svd_c64_run(&factor, a);
    double error = 0.0;
    double norm = 0.0;
    for (size_t j = 0; j < cols; ++j) for (size_t i = 0; i < rows; ++i) {
        double complex value = 0.0;
        for (size_t k = 0; k < rank; ++k) value += u[i + k * rows] * s[k] * conj(v[j + k * cols]);
        const double difference = cabs(value - a[i + j * rows]);
        const double magnitude = cabs(a[i + j * rows]);
        error += difference * difference;
        norm += magnitude * magnitude;
    }
    error = sqrt(error / fmax(norm, 1.0));
    const double left = orthogonality(u, rows, rank);
    const double right = orthogonality(v, cols, rank);
    int valid = status == SVD_OK && error < 1e-10 && left < 1e-10 && right < 1e-10;
    for (size_t i = 1; i < rank; ++i) valid &= s[i - 1] >= s[i] && s[i] >= 0.0;
    memcpy(saved, u, rows * rank * sizeof(*u));
    status = randomized_svd_c64_run(&factor, a);
    valid &= status == SVD_OK && memcmp(saved, u, rows * rank * sizeof(*u)) == 0;
    printf("%zux%zu k=%zu q=%zu zero=%d real=%d error=%.3e U=%.3e V=%.3e %s\n", rows, cols, rank, iterations, zero, real, error, left, right, valid ? "PASS" : "FAIL");
    free(factor.real_work);
    free(factor.work);
    free(s); free(saved); free(v); free(u); free(a);
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}

static int check_scale(double scale) {
    double complex a[36], reference[36], u[36], v[36], work[216];
    double s[6], real_work[12];
    for (size_t i = 0; i < 36; ++i) {
        reference[i] = sin(i + 0.3) + I * cos(0.7 * i + 0.1);
        a[i] = reference[i] * scale;
    }
    svd_c64 exact = {6, 6, u, s, v, work, real_work};
    randomized_svd_c64 randomized = {6, 6, 6, 0, 1, 17, u, s, v, work, real_work};
    int failed = 0;
    for (int algorithm = 0; algorithm < 2; ++algorithm) {
        const svd_status status = algorithm == 0 ? svd_c64_run(&exact, a) : randomized_svd_c64_run(&randomized, a);
        double error = 0.0;
        for (size_t j = 0; j < 6; ++j) for (size_t i = 0; i < 6; ++i) {
            double complex value = 0.0;
            for (size_t k = 0; k < 6; ++k) value += u[i + 6 * k] * (s[k] / scale) * conj(v[j + 6 * k]);
            error = fmax(error, cabs(value - reference[i + 6 * j]));
        }
        failed |= status != SVD_OK || !isfinite(error) || error > 1e-10;
        printf("scale=%.0e algorithm=%d error=%.3e\n", scale, algorithm, error);
    }
    return failed;
}

int main(void) {
    int failed = 0;
    const size_t shapes[][2] = {{24, 24}, {40, 18}, {18, 40}};
    for (size_t shape = 0; shape < 3; ++shape) for (int real = 0; real < 2; ++real) for (size_t q = 0; q <= 2; ++q) {
        failed |= check(shapes[shape][0], shapes[shape][1], 5, q, 0, real);
        failed |= check(shapes[shape][0], shapes[shape][1], 5, q, 1, real);
    }
    failed |= check(18, 40, 18, 1, 0, 0);
    failed |= check(256, 256, 20, 1, 0, 0);
    failed |= check(512, 256, 20, 1, 0, 1);
    failed |= check(256, 512, 20, 1, 0, 0);
    failed |= check_scale(1e-200);
    failed |= check_scale(1e-20);
    failed |= check_scale(1e20);
    failed |= check_scale(1e200);
    double complex value = 1.0;
    randomized_svd_c64 invalid = {0};
    failed |= randomized_svd_c64_run(nullptr, &value) != SVD_BAD_ARGUMENT;
    failed |= randomized_svd_c64_run(&invalid, &value) != SVD_BAD_ARGUMENT;
    invalid.rows = SIZE_MAX;
    invalid.cols = SIZE_MAX;
    invalid.rank = 1;
    failed |= randomized_svd_c64_work_size(&invalid) != 0;
    double complex u[4], v[4], work[4];
    double s[2], e[2];
    svd_c64 bad = {SIZE_MAX, 2, u, s, v, work, e};
    failed |= svd_c64_run(&bad, &value) != SVD_BAD_ARGUMENT;
    bad.rows = 1;
    bad.cols = 2;
    failed |= svd_c64_run(&bad, work) != SVD_BAD_ARGUMENT;
    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
