using LinearAlgebra
using Printf
using Random
using SVDScratch

BLAS.set_num_threads(1)
const ROOT = dirname(@__DIR__)

function measure(function_to_run)
    function_to_run()
    times = Float64[]
    bytes = Int[]
    result = nothing
    for _ in 1:3
        GC.gc()
        sample = @timed function_to_run()
        push!(times, sample.time)
        push!(bytes, sample.bytes)
        result = sample.value
    end
    sort!(times)
    result, times[2], minimum(bytes)
end

function spectrum_matrix(rows, columns, values, rng)
    width = length(values)
    left = Matrix(qr(randn(rng, ComplexF64, rows, width)).Q)[:, 1:width]
    right = Matrix(qr(randn(rng, ComplexF64, columns, width)).Q)[:, 1:width]
    left * Diagonal(values) * right'
end

function compare(name, A, rank, directory, report; image=false)
    workspace = RandomizedWorkspace(A, rank)
    random_rng = MersenneTwister(17)
    random_run() = randomized_svd!(workspace, A, rank; rng=Random.seed!(random_rng, 17))
    F, randomized_time, bytes = measure(random_run)
    exact, exact_time, exact_bytes = measure(() -> golub_reinsch_svd(A))
    reference, lapack_time, _ = measure(() -> svd(A))
    optimal = sqrt(sum(abs2, reference.S[rank + 1:end])) / max(norm(A), eps())
    error = reconstruction_error(A, F)
    @assert F.converged && exact.converged
    @assert orthogonality_error(F.U) < 1e-10 && orthogonality_error(F.V) < 1e-10
    @assert error <= 1.3optimal + 1e-10
    path = joinpath(directory, "$name.bin")
    open(path, "w") do io
        write(io, UInt64[size(A, 1), size(A, 2), rank])
        write(io, A)
    end
    executable = joinpath(ROOT, "c", "bench_randomized")
    c_exact, c_randomized, c_optimal, c_error, c_left, c_right = parse.(Float64, split(read(`$executable $path`, String)))
    @assert isapprox(c_optimal, optimal; atol=1e-10, rtol=1e-8)
    @assert c_left < 1e-10 && c_right < 1e-10
    psnr = image ? -10log10(error^2 * sum(abs2, A) / length(A)) : NaN
    optimal_psnr = image ? -10log10(optimal^2 * sum(abs2, A) / length(A)) : NaN
    @printf("%-18s %4dx%-4d k=%2d Julia %.2f→%.2f ms C %.2f→%.2f ms LAPACK %.2f ms error %.4g/%.4g C %.4g alloc %d/%d bytes\n", name, size(A)..., rank, 1e3exact_time, 1e3randomized_time, 1e3c_exact, 1e3c_randomized, 1e3lapack_time, error, optimal, c_error, bytes, exact_bytes)
    image && @printf("  rank-%d PSNR randomized %.2f dB / optimal %.2f dB\n", rank, psnr, optimal_psnr)
    @printf(report, "%s,%d,%d,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%d,%d,%.9g,%.9g,%.9g,%.6g,%.6g\n", name, size(A)..., rank, exact_time, randomized_time, c_exact, c_randomized, lapack_time, bytes, exact_bytes, optimal, error, c_error, psnr, optimal_psnr)
    flush(report)
    flush(stdout)
    F
end

function main()
    flags = "-std=c23 -O3 -DNDEBUG -march=native -mtune=native -fno-math-errno -fno-trapping-math -ffp-contract=fast -Wall -Wextra -Wpedantic"
    openmp = get(ENV, "OPENMP", "0")
    openmp in ("0", "1") || error("OPENMP must be 0 or 1")
    run(`make -C $(joinpath(ROOT, "c")) -B bench_randomized CFLAGS=$flags OPENMP=$openmp`)
    rng = MersenneTwister(91)
    directory = mktempdir()
    report_path = joinpath(ROOT, "outputs", openmp == "1" ? "randomized_openmp_benchmark.csv" : "randomized_benchmark.csv")
    open(report_path, "w") do report
        println(report, "case,rows,cols,rank,julia_exact_s,julia_randomized_s,c_exact_s,c_randomized_s,lapack_s,julia_randomized_bytes,julia_exact_bytes,optimal_error,julia_error,c_error,julia_psnr,optimal_psnr")
        for (name, rows, columns) in (("decay_square", 128, 128), ("decay_tall", 256, 96), ("decay_wide", 96, 256))
            values = exp.(-range(0, 12; length=min(rows, columns)))
            compare(name, spectrum_matrix(rows, columns, values, rng), 16, directory, report)
        end
        compare("flat_spectrum", randn(rng, ComplexF64, 128, 128), 16, directory, report)
        compare("rank_deficient", spectrum_matrix(96, 64, ones(5), rng), 8, directory, report)
        compare("zero", zeros(ComplexF64, 64, 64), 8, directory, report)
        for name in ("landscape", "city", "lake"), size in (256, 512)
            ppm = joinpath(directory, "$(name)_$(size).ppm")
            source = joinpath(ROOT, "assets", "real_$(name).jpg")
            run(`magick $source -resize $("$(size)x$(size)^") -gravity center -extent $("$(size)x$(size)") -depth 8 $ppm`)
            channels = read_ppm(ppm)
            A = ComplexF64.(channels[1])
            compare("$(name)_$(size)", A, 20, directory, report; image=true)
        end
    end
    println("Saved $report_path; median of three warm runs, BLAS threads=1, complex Float64, p=8, q=1. Photo benchmarks use the red channel; timing excludes reconstruction and I/O.")
    println("OpenMP=$openmp, OMP_NUM_THREADS=$(get(ENV, "OMP_NUM_THREADS", "runtime default")), OMP_WAIT_POLICY=$(get(ENV, "OMP_WAIT_POLICY", "runtime default"))")
    println("OMP_PROC_BIND=$(get(ENV, "OMP_PROC_BIND", "runtime default")), OMP_PLACES=$(get(ENV, "OMP_PLACES", "runtime default"))")
end

main()
