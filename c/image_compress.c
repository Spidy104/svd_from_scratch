#include "svd.h"

#include <complex.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    size_t width;
    size_t height;
    uint8_t *pixels;
} image;

static int token(FILE *stream, char *buffer, size_t capacity) {
    int character;
    do {
        character = fgetc(stream);
        if (character == '#') {
            while (character != '\n' && character != EOF) character = fgetc(stream);
        }
    } while (isspace(character) || character == '#');
    if (character == EOF) return 0;
    size_t length = 0;
    do {
        if (length + 1 >= capacity) return 0;
        buffer[length++] = (char)character;
        character = fgetc(stream);
    } while (!isspace(character) && character != EOF);
    buffer[length] = '\0';
    if (character == '\r') {
        const int next = fgetc(stream);
        if (next != '\n' && next != EOF) ungetc(next, stream);
    }
    return 1;
}

static int dimension(FILE *stream, size_t *output) {
    char value[32];
    if (!token(stream, value, sizeof(value)) || !isdigit((unsigned char)value[0])) return 0;
    char *end;
    errno = 0;
    const unsigned long long number = strtoull(value, &end, 10);
    if (errno != 0 || *end != '\0' || number == 0 || number > SIZE_MAX) return 0;
    *output = (size_t)number;
    return 1;
}

static int read_ppm(const char *path, image *output) {
    FILE *stream = fopen(path, "rb");
    char value[32];
    if (stream == nullptr) return 0;
    const int valid = token(stream, value, sizeof(value)) && strcmp(value, "P6") == 0 && dimension(stream, &output->width) && dimension(stream, &output->height) && token(stream, value, sizeof(value)) && strcmp(value, "255") == 0;
    if (!valid) {
        fclose(stream);
        return 0;
    }
    if (output->height > SIZE_MAX / 3 || output->width > SIZE_MAX / (3 * output->height) || output->width > SIZE_MAX / sizeof(double complex) / output->height) {
        fclose(stream);
        return 0;
    }
    const size_t bytes = 3 * output->width * output->height;
    output->pixels = malloc(bytes);
    const int complete = output->pixels != nullptr && fread(output->pixels, 1, bytes, stream) == bytes;
    fclose(stream);
    if (!complete) {
        free(output->pixels);
        output->pixels = nullptr;
    }
    return complete;
}

static int write_ppm(const char *path, const image *input) {
    FILE *stream = fopen(path, "wb");
    if (stream == nullptr) return 0;
    const int complete = fprintf(stream, "P6\n%zu %zu\n255\n", input->width, input->height) > 0 && fwrite(input->pixels, 1, 3 * input->width * input->height, stream) == 3 * input->width * input->height;
    const int closed = fclose(stream) == 0;
    return complete && closed;
}

static size_t at(size_t rows, size_t row, size_t column) {
    return row + column * rows;
}

static uint8_t byte_of(double value) {
    const double clipped = fmin(fmax(value, 0.0), 1.0);
    return (uint8_t)llround(255.0 * clipped);
}

int main(int argc, char **argv) {
    const char *input_path = argc > 1 ? argv[1] : "../assets/real_landscape.ppm";
    const char *output_directory = argc > 2 ? argv[2] : "../outputs";
    const size_t ranks[] = {5, 20, 50};
    const size_t rank_count = sizeof(ranks) / sizeof(*ranks);
    image original = {0};
    if (!read_ppm(input_path, &original)) {
        fprintf(stderr, "cannot read %s\n", input_path);
        return EXIT_FAILURE;
    }
    const size_t rows = original.height;
    const size_t columns = original.width;
    const size_t modes = rows < columns ? rows : columns;
    if (ranks[rank_count - 1] > modes) {
        fprintf(stderr, "image is too small for rank %zu\n", ranks[rank_count - 1]);
        free(original.pixels);
        return EXIT_FAILURE;
    }
    const size_t samples = rows * columns;
    double complex *matrix = malloc(samples * sizeof(*matrix));
    double complex *u = malloc(rows * modes * sizeof(*u));
    double complex *v = malloc(columns * modes * sizeof(*v));
    double complex *work = malloc(samples * sizeof(*work));
    double *s = malloc(modes * sizeof(*s));
    double *superdiagonal = malloc(modes * sizeof(*superdiagonal));
    int result = EXIT_FAILURE;
    image outputs[3] = {0};
    for (size_t i = 0; i < rank_count; ++i) {
        outputs[i].width = columns;
        outputs[i].height = rows;
        outputs[i].pixels = malloc(3 * samples);
    }
    if (matrix == nullptr || u == nullptr || v == nullptr || work == nullptr || s == nullptr || superdiagonal == nullptr || outputs[0].pixels == nullptr || outputs[1].pixels == nullptr || outputs[2].pixels == nullptr) {
        fprintf(stderr, "allocation failed\n");
        goto cleanup;
    }
    double errors[3] = {0};
    svd_c64 factor = {rows, columns, u, s, v, work, superdiagonal};
    for (size_t channel = 0; channel < 3; ++channel) {
        for (size_t column = 0; column < columns; ++column) for (size_t row = 0; row < rows; ++row) {
            const size_t pixel = 3 * (row * columns + column) + channel;
            matrix[at(rows, row, column)] = original.pixels[pixel] / 255.0;
        }
        if (svd_c64_run(&factor, matrix) != SVD_OK) {
            fprintf(stderr, "SVD did not converge for RGB channel %zu\n", channel);
            goto cleanup;
        }
        for (size_t choice = 0; choice < rank_count; ++choice) for (size_t column = 0; column < columns; ++column) for (size_t row = 0; row < rows; ++row) {
            double value = 0.0;
            for (size_t mode = 0; mode < ranks[choice]; ++mode) value += creal(u[at(rows, row, mode)] * s[mode] * conj(v[at(columns, column, mode)]));
            const size_t pixel = 3 * (row * columns + column) + channel;
            const double difference = original.pixels[pixel] / 255.0 - value;
            errors[choice] += difference * difference;
            outputs[choice].pixels[pixel] = byte_of(value);
        }
    }
    for (size_t choice = 0; choice < rank_count; ++choice) {
        char path[512];
        const int length = snprintf(path, sizeof(path), "%s/landscape_c_rank_%zu.ppm", output_directory, ranks[choice]);
        const double mse = errors[choice] / (3.0 * samples);
        if (length < 0 || (size_t)length >= sizeof(path) || !write_ppm(path, &outputs[choice])) {
            fprintf(stderr, "cannot write %s\n", path);
            goto cleanup;
        }
        printf("rank=%zu MSE=%.6e PSNR=%.2f dB scalar_ratio=%.2fx output=%s\n", ranks[choice], mse, mse == 0.0 ? INFINITY : 10.0 * log10(1.0 / mse), (double)samples / (ranks[choice] * (rows + columns + 1)), path);
    }
    result = EXIT_SUCCESS;
cleanup:
    for (size_t i = 0; i < rank_count; ++i) free(outputs[i].pixels);
    free(superdiagonal); free(s); free(work); free(v); free(u); free(matrix); free(original.pixels);
    return result;
}
