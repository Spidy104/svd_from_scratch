# SVD from scratch

Singular value decomposition in Julia and C23. Includes exact and randomized methods, tests, and a few applications.

## Implementations

- Julia: one-sided Jacobi and Golub–Reinsch SVD for real and complex matrices.
- C: complex double-precision Golub–Reinsch SVD with caller-owned buffers.
- Randomized SVD in both languages, with oversampling and power iterations.
- Optional OpenMP for the C randomized matrix products.

Tests cover square and rectangular matrices, rank deficiency, repeated singular values, zero matrices, and very small or large input scales. Reconstruction and orthogonality are checked against Julia's LAPACK-backed SVD.

## Requirements

Julia 1.13, a compiler supporting `-std=c23`, and Make. ImageMagick (`magick`) is needed for scripts that convert JPEG images. OpenMP requires compiler and runtime support.

The Julia package uses standard-library dependencies only.

## Run

From the repository root:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=test test/runtests.jl
make -C c test
```

Julia benchmarks and examples:

```sh
julia --project=scripts scripts/bench.jl
julia --project=scripts scripts/randomized_bench.jl
julia --project=scripts scripts/mimo.jl
julia --project=scripts scripts/image_compress.jl
```

The randomized benchmark builds its C executable and compares Julia, C, and LAPACK on the same inputs. Running it replaces the corresponding CSV in `outputs/`.

To enable OpenMP for that benchmark:

```sh
OPENMP=1 OMP_NUM_THREADS=4 OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=PASSIVE julia --project=scripts scripts/randomized_bench.jl
```

C sanitizer tests:

```sh
make -C c debug
```

## Applications

The MIMO example measures QPSK bit error rate, channel capacity, and stream selection on simulated Rayleigh channels. Image compression reconstructs RGB channels at selected ranks. These are numerical examples, not a deployed communications system or an image codec.

## Benchmarks

Saved measurements are in `outputs/`. The randomized comparison uses the median of three warmed runs, one BLAS thread, and `ComplexF64` inputs. Image cases use the red channel. Timings exclude image conversion and reconstruction.

One saved 512×512 landscape case, targeting rank 20:

| Method | Time | Relative reconstruction error |
| --- | ---: | ---: |
| Julia randomized SVD | 11.19 ms | 0.15895 |
| LAPACK full SVD, then rank-20 truncation | 139.36 ms | 0.15620 |
| Julia exact SVD | 876.82 ms | — |

The randomized method computes an approximation and less output than a full SVD. This is not an equivalent-work speed comparison. The exact implementation did not beat LAPACK in this case. Results depend on matrix shape, spectrum, hardware, and threading.

## Layout

`src/` contains the Julia package, `c/` the C implementation, `test/` correctness tests, and `scripts/` benchmarks and examples. Input images are in `assets/`; saved results are in `outputs/`.
