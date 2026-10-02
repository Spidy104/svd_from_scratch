using LinearAlgebra
using Printf
using Random
using SVDScratch

function measure(f)
    f()
    elapsed = @elapsed for _ in 1:10_000
        f()
    end
    allocated = @allocated f()
    elapsed / 10_000, allocated
end

rng = MersenneTwister(107)
println("square complex MIMO factorization")
println("size | QR-SVD ns | Jacobi ns | LAPACK ns | auto")
println("-----------------------------------------------")
for n in (2, 4, 8, 16)
    H = Matrix{ComplexF64}(undef, n, n)
    rayleigh_channel!(H, rng)
    qr_time, _ = measure(() -> svd_scratch(H; algorithm=:golub_reinsch))
    jacobi_time, _ = measure(() -> svd_scratch(H; algorithm=:jacobi))
    lapack_time, _ = measure(() -> svd(H; full=false))
    auto = typeof(svd_scratch(H)) === JacobiSVD{ComplexF64} ? "Jacobi" : "QR"
    @printf("%4dx%-2d | %9.1f | %9.1f | %9.1f | %s\n", n, n, 1e9 * qr_time, 1e9 * jacobi_time, 1e9 * lapack_time, auto)
end
