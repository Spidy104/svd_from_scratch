#include "svd.h"

#include <complex.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static inline size_t index_of(size_t rows, size_t row, size_t column) {
    return row + column * rows;
}

static inline double complex phase_of(double complex value) {
    const double magnitude = cabs(value);
    return magnitude == 0.0 ? 1.0 : value / magnitude;
}

static inline double abs2_of(double complex value) {
    return creal(value) * creal(value) + cimag(value) * cimag(value);
}

static void identity(double complex *matrix, size_t rows, size_t columns) {
    memset(matrix, 0, rows * columns * sizeof(*matrix));
    for (size_t j = 0; j < columns; ++j) {
        matrix[index_of(rows, j, j)] = 1.0;
    }
}

static double left_reflector(double complex *matrix, size_t rows, size_t row, size_t column) {
    const size_t pivot = index_of(rows, row, column);
    const double complex alpha = matrix[pivot];
    double tail_norm2 = 0.0;
    for (size_t i = row + 1; i < rows; ++i) {
        tail_norm2 += abs2_of(matrix[index_of(rows, i, column)]);
    }
    const double norm = hypot(cabs(alpha), sqrt(tail_norm2));
    if (norm == 0.0) {
        return 0.0;
    }
    const double complex beta = -phase_of(alpha) * norm;
    const double complex denominator = alpha - beta;
    for (size_t i = row + 1; i < rows; ++i) {
        matrix[index_of(rows, i, column)] /= denominator;
    }
    matrix[pivot] = beta;
    return creal((beta - alpha) / beta);
}

static double right_reflector(double complex *matrix, size_t rows, size_t columns, size_t row, size_t column) {
    const size_t pivot = index_of(rows, row, column);
    const double complex alpha = matrix[pivot];
    double tail_norm2 = 0.0;
    for (size_t j = column + 1; j < columns; ++j) {
        tail_norm2 += abs2_of(matrix[index_of(rows, row, j)]);
    }
    const double norm = hypot(cabs(alpha), sqrt(tail_norm2));
    if (norm == 0.0) {
        return 0.0;
    }
    const double complex beta = -phase_of(alpha) * norm;
    const double complex denominator = alpha - beta;
    for (size_t j = column + 1; j < columns; ++j) {
        matrix[index_of(rows, row, j)] /= denominator;
    }
    matrix[pivot] = beta;
    return creal((beta - alpha) / beta);
}

static void apply_left(double complex *target, size_t target_rows, size_t target_columns, const double complex *storage, size_t storage_rows, double tau, size_t row, size_t vector_column, size_t first_column) {
    if (tau == 0.0) {
        return;
    }
    for (size_t j = first_column; j < target_columns; ++j) {
        double complex projection = target[index_of(target_rows, row, j)];
        for (size_t i = row + 1; i < target_rows; ++i) {
            projection += conj(storage[index_of(storage_rows, i, vector_column)]) * target[index_of(target_rows, i, j)];
        }
        projection *= tau;
        target[index_of(target_rows, row, j)] -= projection;
        for (size_t i = row + 1; i < target_rows; ++i) {
            target[index_of(target_rows, i, j)] -= storage[index_of(storage_rows, i, vector_column)] * projection;
        }
    }
}

static void apply_right(double complex *target, size_t target_rows, size_t target_columns, const double complex *storage, size_t storage_rows, double tau, size_t vector_row, size_t column, size_t first_row) {
    if (tau == 0.0) {
        return;
    }
    for (size_t i = first_row; i < target_rows; ++i) {
        double complex projection = target[index_of(target_rows, i, column)];
        for (size_t j = column + 1; j < target_columns; ++j) {
            projection += target[index_of(target_rows, i, j)] * conj(storage[index_of(storage_rows, vector_row, j)]);
        }
        projection *= tau;
        target[index_of(target_rows, i, column)] -= projection;
        for (size_t j = column + 1; j < target_columns; ++j) {
            target[index_of(target_rows, i, j)] -= projection * storage[index_of(storage_rows, vector_row, j)];
        }
    }
}

static void rotate_columns(double complex *matrix, size_t rows, size_t first, size_t second, double c, double s) {
    for (size_t i = 0; i < rows; ++i) {
        const double complex a = matrix[index_of(rows, i, first)];
        const double complex b = matrix[index_of(rows, i, second)];
        matrix[index_of(rows, i, first)] = c * a + s * b;
        matrix[index_of(rows, i, second)] = c * b - s * a;
    }
}

static void sort_singular_values(svd_c64 *factor) {
    const size_t n = factor->cols;
    for (size_t i = 0; i < n; ++i) {
        size_t best = i;
        for (size_t j = i + 1; j < n; ++j) {
            if (factor->s[j] > factor->s[best]) {
                best = j;
            }
        }
        if (best == i) {
            continue;
        }
        const double singular = factor->s[i];
        factor->s[i] = factor->s[best];
        factor->s[best] = singular;
        for (size_t row = 0; row < factor->rows; ++row) {
            const size_t left = index_of(factor->rows, row, i);
            const size_t right = index_of(factor->rows, row, best);
            const double complex value = factor->u[left];
            factor->u[left] = factor->u[right];
            factor->u[right] = value;
        }
        for (size_t row = 0; row < n; ++row) {
            const size_t left = index_of(n, row, i);
            const size_t right = index_of(n, row, best);
            const double complex value = factor->v[left];
            factor->v[left] = factor->v[right];
            factor->v[right] = value;
        }
    }
}

static svd_status bidiagonal_qr(svd_c64 *factor) {
    const size_t n = factor->cols;
    double scale = 0.0;
    for (size_t i = 0; i < n; ++i) {
        scale += fabs(factor->s[i]) + fabs(factor->superdiagonal[i]);
    }
    const double threshold = DBL_EPSILON * scale;
    for (size_t reverse = n; reverse > 0; --reverse) {
        const size_t k = reverse - 1;
        int converged = 0;
        for (size_t iteration = 0; iteration < 128; ++iteration) {
            int flag = 1;
            size_t l = k;
            for (size_t candidate = k + 1; candidate-- > 0;) {
                l = candidate;
                if (candidate == 0 || fabs(factor->superdiagonal[candidate]) <= threshold) {
                    flag = 0;
                    break;
                }
                if (fabs(factor->s[candidate - 1]) <= threshold) {
                    break;
                }
            }
            if (flag) {
                const size_t nm = l - 1;
                double c = 0.0;
                double s = 1.0;
                for (size_t i = l; i <= k; ++i) {
                    const double f = s * factor->superdiagonal[i];
                    factor->superdiagonal[i] = c * factor->superdiagonal[i];
                    if (fabs(f) <= threshold) {
                        break;
                    }
                    const double g = factor->s[i];
                    const double h = hypot(f, g);
                    factor->s[i] = h;
                    c = g / h;
                    s = -f / h;
                    rotate_columns(factor->u, factor->rows, nm, i, c, s);
                }
            }
            double z = factor->s[k];
            if (l == k) {
                if (z < 0.0) {
                    factor->s[k] = -z;
                    for (size_t i = 0; i < n; ++i) {
                        factor->v[index_of(n, i, k)] = -factor->v[index_of(n, i, k)];
                    }
                }
                converged = 1;
                break;
            }
            const size_t nm = k - 1;
            double x = factor->s[l];
            double y = factor->s[nm];
            double g = factor->superdiagonal[nm];
            double h = factor->superdiagonal[k];
            const double shift = h == 0.0 || y == 0.0 ? 0.0 : ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
            const double root = hypot(shift, 1.0);
            const double denominator = shift + copysign(root, shift);
            double f = x == 0.0 || denominator == 0.0 ? 0.0 : ((x - z) * (x + z) + h * (y / denominator - h)) / x;
            double c = 1.0;
            double s = 1.0;
            for (size_t j = l; j <= nm; ++j) {
                const size_t i = j + 1;
                g = factor->superdiagonal[i];
                y = factor->s[i];
                h = s * g;
                g = c * g;
                z = hypot(f, h);
                factor->superdiagonal[j] = z;
                c = z == 0.0 ? 1.0 : f / z;
                s = z == 0.0 ? 0.0 : h / z;
                f = x * c + g * s;
                g = g * c - x * s;
                h = y * s;
                y *= c;
                rotate_columns(factor->v, n, j, i, c, s);
                z = hypot(f, h);
                factor->s[j] = z;
                c = z == 0.0 ? 1.0 : f / z;
                s = z == 0.0 ? 0.0 : h / z;
                f = c * g + s * y;
                x = c * y - s * g;
                rotate_columns(factor->u, factor->rows, j, i, c, s);
            }
            factor->superdiagonal[l] = 0.0;
            factor->superdiagonal[k] = f;
            factor->s[k] = x;
        }
        if (!converged) {
            return SVD_NO_CONVERGENCE;
        }
    }
    sort_singular_values(factor);
    return SVD_OK;
}

static svd_status svd_c64_tall_run(svd_c64 *factor, const double complex *matrix) {
    const size_t rows = factor->rows;
    const size_t columns = factor->cols;
    double complex *const storage = factor->work;
    if (matrix != storage) {
        memcpy(storage, matrix, rows * columns * sizeof(*storage));
    }
    double input_scale = 0.0;
    for (size_t i = 0; i < rows * columns; ++i) {
        const double magnitude = cabs(storage[i]);
        if (!isfinite(magnitude)) return SVD_BAD_ARGUMENT;
        input_scale = fmax(input_scale, magnitude);
    }
    if (input_scale == 0.0) input_scale = 1.0;
    for (size_t i = 0; i < rows * columns; ++i) storage[i] /= input_scale;
    double *const left_tau = factor->s;
    double *const right_tau = factor->superdiagonal;
    for (size_t j = 0; j < columns; ++j) {
        left_tau[j] = left_reflector(storage, rows, j, j);
        apply_left(storage, rows, columns, storage, rows, left_tau[j], j, j, j + 1);
        if (j + 1 < columns) {
            right_tau[j] = right_reflector(storage, rows, columns, j, j + 1);
            apply_right(storage, rows, columns, storage, rows, right_tau[j], j, j + 1, j + 1);
        }
    }
    identity(factor->u, rows, columns);
    for (size_t reverse = columns; reverse > 0; --reverse) {
        const size_t row = reverse - 1;
        apply_left(factor->u, rows, columns, storage, rows, left_tau[row], row, row, 0);
    }
    identity(factor->v, columns, columns);
    for (size_t reverse = columns > 0 ? columns - 1 : 0; reverse > 0; --reverse) {
        const size_t row = reverse - 1;
        const size_t start = row + 1;
        const double tau = right_tau[row];
        for (size_t k = 0; k < columns; ++k) {
            double complex projection = factor->v[index_of(columns, start, k)];
            for (size_t j = start + 1; j < columns; ++j) {
                projection += storage[index_of(rows, row, j)] * factor->v[index_of(columns, j, k)];
            }
            projection *= tau;
            factor->v[index_of(columns, start, k)] -= projection;
            for (size_t j = start + 1; j < columns; ++j) {
                factor->v[index_of(columns, j, k)] -= conj(storage[index_of(rows, row, j)]) * projection;
            }
        }
    }
    double complex right_phase = 1.0;
    for (size_t j = 0; j < columns; ++j) {
        const double complex diagonal = storage[index_of(rows, j, j)] * right_phase;
        const double complex left_phase = phase_of(diagonal);
        factor->s[j] = cabs(diagonal);
        for (size_t i = 0; i < rows; ++i) {
            factor->u[index_of(rows, i, j)] *= left_phase;
        }
        for (size_t i = 0; i < columns; ++i) {
            factor->v[index_of(columns, i, j)] *= right_phase;
        }
        if (j + 1 < columns) {
            const double complex super = conj(left_phase) * storage[index_of(rows, j, j + 1)];
            right_phase = conj(phase_of(super));
            factor->superdiagonal[j + 1] = cabs(super);
        }
    }
    factor->superdiagonal[0] = 0.0;
    const svd_status status = bidiagonal_qr(factor);
    for (size_t j = 0; j < columns; ++j) factor->s[j] *= input_scale;
    return status;
}

svd_status svd_c64_run(svd_c64 *factor, const double complex *matrix) {
    if (factor == nullptr || matrix == nullptr || factor->u == nullptr || factor->s == nullptr || factor->v == nullptr || factor->work == nullptr || factor->superdiagonal == nullptr || factor->rows == 0 || factor->cols == 0) {
        return SVD_BAD_ARGUMENT;
    }
    if (factor->rows > SIZE_MAX / factor->cols / sizeof(*matrix) || (factor->rows < factor->cols && matrix == factor->work)) return SVD_BAD_ARGUMENT;
    if (factor->rows >= factor->cols) {
        return svd_c64_tall_run(factor, matrix);
    }
    for (size_t j = 0; j < factor->cols; ++j) {
        for (size_t i = 0; i < factor->rows; ++i) {
            factor->work[index_of(factor->cols, j, i)] = conj(matrix[index_of(factor->rows, i, j)]);
        }
    }
    svd_c64 transposed = {
        factor->cols,
        factor->rows,
        factor->v,
        factor->s,
        factor->u,
        factor->work,
        factor->superdiagonal,
    };
    return svd_c64_tall_run(&transposed, factor->work);
}
