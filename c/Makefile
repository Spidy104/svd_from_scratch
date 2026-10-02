CC ?= cc
CFLAGS ?= -std=c23 -O3 -DNDEBUG -Wall -Wextra -Wpedantic
LDFLAGS ?= -lm
OPENMP ?= 0

OMPFLAGS = $(if $(filter 1,$(OPENMP)),-fopenmp,)

.PHONY: test_svd bench_svd image_compress bench_randomized test_randomized test bench native debug clean

test_svd: src/svd.c test/test_svd.c include/svd.h
	$(CC) $(CFLAGS) $(OMPFLAGS) -Iinclude src/svd.c test/test_svd.c $(LDFLAGS) -o $@

bench_svd: src/svd.c bench/bench_svd.c include/svd.h
	$(CC) $(CFLAGS) $(OMPFLAGS) -Iinclude src/svd.c bench/bench_svd.c $(LDFLAGS) -o $@

image_compress: src/svd.c image_compress.c include/svd.h
	$(CC) $(CFLAGS) $(OMPFLAGS) -Iinclude src/svd.c image_compress.c $(LDFLAGS) -o $@

bench_randomized: src/svd.c src/randomized_svd.c bench/bench_randomized.c include/svd.h include/randomized_svd.h
	$(CC) $(CFLAGS) $(OMPFLAGS) -Iinclude src/svd.c src/randomized_svd.c bench/bench_randomized.c $(LDFLAGS) -o $@

test_randomized: src/svd.c src/randomized_svd.c test/test_randomized.c include/svd.h include/randomized_svd.h
	$(CC) $(CFLAGS) $(OMPFLAGS) -Iinclude src/svd.c src/randomized_svd.c test/test_randomized.c $(LDFLAGS) -o $@

test: test_svd test_randomized
	./test_svd
	./test_randomized

bench: bench_svd
	./bench_svd 64

native: override CFLAGS += -march=native -mtune=native -fno-math-errno -fno-trapping-math -ffp-contract=fast
native: bench

debug: override CFLAGS = -std=c23 -Og -g3 -Wall -Wextra -Wpedantic -fsanitize=address,undefined
debug: test_svd test_randomized
	ASAN_OPTIONS=detect_leaks=0 ./test_svd
	ASAN_OPTIONS=detect_leaks=0 ./test_randomized

clean:
	rm -f test_svd bench_svd image_compress bench_randomized test_randomized
