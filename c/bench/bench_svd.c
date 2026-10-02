#include "svd.h"

#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static unsigned long long state = 29;

static double sample(void) {
    state ^= state << 7;
    state ^= state >> 9;
    return ((double)(state & 0xffff) / 32768.0) - 1.0;
}

static double now(void) {
    struct timespec value;
    timespec_get(&value, TIME_MONOTONIC);
    return (double)value.tv_sec + 1e-9 * (double)value.tv_nsec;
}

int main(int argc, char **argv) {
    const size_t rows = argc > 1 ? strtoull(argv[1], NULL, 10) : 64;
    const size_t columns = argc > 2 ? strtoull(argv[2], NULL, 10) : rows;
    const size_t density_percent = argc > 3 ? strtoull(argv[3], NULL, 10) : 100;
    const size_t modes = rows < columns ? rows : columns;
    const size_t count = rows * columns;
    double complex *a = malloc(count * sizeof(*a));
    double complex *u = malloc(rows * modes * sizeof(*u));
    double complex *v = malloc(columns * modes * sizeof(*v));
    double complex *work = malloc(count * sizeof(*work));
    double *s = malloc(modes * sizeof(*s));
    double *e = malloc(modes * sizeof(*e));
    if (!a || !u || !v || !work || !s || !e) return EXIT_FAILURE;
    for (size_t i = 0; i < count; ++i) a[i] = 50.0 * (sample() + 1.0) < density_percent ? sample() + sample() * I : 0.0;
    svd_c64 f = {rows, columns, u, s, v, work, e};
    if (svd_c64_run(&f, a) != SVD_OK) return EXIT_FAILURE;
    double best = 1e9;
    for (size_t run = 0; run < 7; ++run) {
        const double start = now();
        if (svd_c64_run(&f, a) != SVD_OK) return EXIT_FAILURE;
        const double elapsed = now() - start;
        if (elapsed < best) best = elapsed;
    }
    printf("%zux%zu density=%zu%% %.6f ms\n", rows, columns, density_percent, 1e3 * best);
    free(e); free(s); free(work); free(v); free(u); free(a);
}
