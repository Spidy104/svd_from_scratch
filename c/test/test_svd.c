#include "svd.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned long long state = 17;

static double sample(void) {
    state ^= state << 7;
    state ^= state >> 9;
    return ((double)(state & 0xffff) / 32768.0) - 1.0;
}

static size_t at(size_t rows, size_t row, size_t column) {
    return row + column * rows;
}

static double frobenius(const double complex *matrix, size_t rows, size_t columns) {
    double total = 0.0;
    for (size_t i = 0; i < rows * columns; ++i) total += creal(matrix[i]) * creal(matrix[i]) + cimag(matrix[i]) * cimag(matrix[i]);
    return sqrt(total);
}

static double reconstruction_error(const double complex *a, const svd_c64 *f) {
    const size_t rank = f->rows < f->cols ? f->rows : f->cols;
    double complex *r = calloc(f->rows * f->cols, sizeof(*r));
    for (size_t j = 0; j < f->cols; ++j) for (size_t k = 0; k < rank; ++k) for (size_t i = 0; i < f->rows; ++i) r[at(f->rows, i, j)] += f->u[at(f->rows, i, k)] * f->s[k] * conj(f->v[at(f->cols, j, k)]);
    for (size_t i = 0; i < f->rows * f->cols; ++i) r[i] -= a[i];
    const double error = frobenius(r, f->rows, f->cols) / fmax(frobenius(a, f->rows, f->cols), 1.0);
    free(r);
    return error;
}

static double orthogonality_error(const double complex *q, size_t rows, size_t columns) {
    double total = 0.0;
    for (size_t j = 0; j < columns; ++j) for (size_t k = 0; k < columns; ++k) {
        double complex value = j == k ? -1.0 : 0.0;
        for (size_t i = 0; i < rows; ++i) value += conj(q[at(rows, i, j)]) * q[at(rows, i, k)];
        total += creal(value) * creal(value) + cimag(value) * cimag(value);
    }
    return sqrt(total);
}

static int run_case(const char *name, size_t rows, size_t columns, size_t rank, int zero) {
    const size_t modes = rows < columns ? rows : columns;
    double complex *a = calloc(rows * columns, sizeof(*a));
    double complex *u = malloc(rows * modes * sizeof(*u));
    double complex *v = malloc(columns * modes * sizeof(*v));
    double complex *work = malloc(rows * columns * sizeof(*work));
    double *s = malloc(modes * sizeof(*s));
    double *e = malloc(modes * sizeof(*e));
    if (!a || !u || !v || !work || !s || !e) return EXIT_FAILURE;
    if (!zero && rank == modes) for (size_t i = 0; i < rows * columns; ++i) a[i] = sample() + sample() * I;
    if (!zero && rank < modes) {
        double complex *left = malloc(rows * rank * sizeof(*left));
        double complex *right = malloc(rank * columns * sizeof(*right));
        if (!left || !right) return EXIT_FAILURE;
        for (size_t i = 0; i < rows * rank; ++i) left[i] = sample() + sample() * I;
        for (size_t i = 0; i < rank * columns; ++i) right[i] = sample() + sample() * I;
        for (size_t j = 0; j < columns; ++j) for (size_t k = 0; k < rank; ++k) for (size_t i = 0; i < rows; ++i) a[at(rows, i, j)] += left[at(rows, i, k)] * right[at(rank, k, j)];
        free(right); free(left);
    }
    svd_c64 f = {rows, columns, u, s, v, work, e};
    const svd_status status = svd_c64_run(&f, a);
    const double reconstruction = reconstruction_error(a, &f);
    const double left_orthogonality = orthogonality_error(u, rows, modes);
    const double right_orthogonality = orthogonality_error(v, columns, modes);
    printf("%-16s status=%d reconstruction=%.3e U=%.3e V=%.3e\n", name, status, reconstruction, left_orthogonality, right_orthogonality);
    free(e); free(s); free(work); free(v); free(u); free(a);
    return status == SVD_OK && reconstruction < 1e-10 && left_orthogonality < 1e-10 && right_orthogonality < 1e-10 ? EXIT_SUCCESS : EXIT_FAILURE;
}

static int run_ill_conditioned(void) {
    const size_t n = 6;
    const double values[6] = {1.0, 1e-3, 1e-6, 1e-9, 1e-12, 1e-14};
    double complex a[36] = {0}, u[36], v[36], work[36];
    double s[6], e[6];
    for (size_t i = 0; i < n; ++i) a[at(n, i, i)] = values[i] * (cos((double)i) + sin((double)i) * I);
    svd_c64 f = {n, n, u, s, v, work, e};
    const svd_status status = svd_c64_run(&f, a);
    const double reconstruction = reconstruction_error(a, &f);
    const double left_orthogonality = orthogonality_error(u, n, n);
    const double right_orthogonality = orthogonality_error(v, n, n);
    printf("%-16s status=%d reconstruction=%.3e U=%.3e V=%.3e\n", "ill conditioned", status, reconstruction, left_orthogonality, right_orthogonality);
    return status == SVD_OK && reconstruction < 1e-10 && left_orthogonality < 1e-10 && right_orthogonality < 1e-10 ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(void) {
    int failed = 0;
    failed |= run_case("MIMO 4x4", 4, 4, 4, 0);
    failed |= run_case("MIMO tall", 12, 6, 6, 0);
    failed |= run_case("MIMO wide", 6, 12, 6, 0);
    failed |= run_case("rank deficient", 12, 7, 3, 0);
    failed |= run_ill_conditioned();
    failed |= run_case("zero", 9, 4, 0, 1);
    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
