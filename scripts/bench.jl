using LinearAlgebra
using Printf
using Random
using SVDScratch

function measure(f, repeats)
    f()
    best = Inf
    for _ in 1:repeats
        elapsed = @elapsed f()
        best = min(best, elapsed)
    end
    best, @allocated f()
end

function report(label, A)
    jacobi_time, jacobi_bytes = measure(() -> jacobi_svd(A), 3)
    qr_time, qr_bytes = measure(() -> golub_reinsch_svd(A), 3)
    lapack_time, lapack_bytes = measure(() -> svd(A; full=false), 3)
    @printf("%-14s %10.4f %12d %10.4f %12d %10.4f %12d %8.2fx\n", label, 1e3 * jacobi_time, jacobi_bytes, 1e3 * qr_time, qr_bytes, 1e3 * lapack_time, lapack_bytes, qr_time / lapack_time)
end

rng = MersenneTwister(11)
println("shape         Jacobi ms Jacobi bytes  QR-SVD ms QR-SVD bytes  LAPACK ms LAPACK bytes QR/LAPACK")
println("----------------------------------------------------------------------------------------------")
report("16x16", randn(rng, 16, 16))
report("64x64", randn(rng, 64, 64))
report("128x32", randn(rng, 128, 32))

println()
println("complex MIMO channels")
println("shape         Jacobi ms Jacobi bytes  QR-SVD ms QR-SVD bytes  LAPACK ms LAPACK bytes QR/LAPACK")
println("----------------------------------------------------------------------------------------------")
report("16x16", randn(rng, ComplexF64, 16, 16))
report("64x64", randn(rng, ComplexF64, 64, 64))
report("128x32", randn(rng, ComplexF64, 128, 32))
