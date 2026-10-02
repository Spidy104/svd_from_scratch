using LinearAlgebra
using Random
using Test
using SVDScratch

@testset "Scale and weak-mode regressions" begin
    rng = MersenneTwister(241)
    for T in (Float64, ComplexF64), (rows, columns) in ((8, 8), (12, 6), (6, 12))
        A = randn(rng, T, rows, columns)
        for scale in (1e-200, 1e-20, 1e20, 1e200)
            for F in (jacobi_svd(scale * A), golub_reinsch_svd(scale * A), randomized_svd(scale * A, min(rows, columns); rng=MersenneTwister(17)))
                @test F.converged
                @test norm(A - F.U * Diagonal(F.S ./ scale) * adjoint(F.V)) / norm(A) < 1e-10
                @test orthogonality_error(F.U) < 1e-10
                @test orthogonality_error(F.V) < 1e-10
            end
        end
        Q = Matrix(qr(randn(rng, T, rows, 6)).Q)[:, 1:6]
        V = Matrix(qr(randn(rng, T, columns, 6)).Q)[:, 1:6]
        A = Q * Diagonal([1.0, 0.1, 1e-6, 1e-9, 1e-12, 1e-14]) * adjoint(V)
        @test reconstruction_error(A, jacobi_svd(A)) < 1e-12
    end
    @test_throws ArgumentError golub_reinsch_svd(fill(NaN, 4, 4))
    @test_throws ArgumentError jacobi_svd(fill(Inf, 4, 4))
    @test_throws ArgumentError randomized_svd(fill(NaN, 4, 4), 2)
end

@testset "Randomized SVD" begin
    rng = MersenneTwister(211)
    for T in (Float64, ComplexF64), (rows, columns) in ((24, 24), (40, 18), (18, 40))
        A = randn(rng, T, rows, columns)
        rank = 5
        factor = randomized_svd(A, rank; power_iterations=2, rng=MersenneTwister(17))
        optimal = sqrt(sum(abs2, svdvals(A)[rank + 1:end])) / norm(A)
        @test factor.converged
        @test issorted(factor.S; rev=true)
        @test size(factor.U) == (rows, rank)
        @test size(factor.V) == (columns, rank)
        @test optimal * (1 - 1e-12) <= reconstruction_error(A, factor) <= 1.1optimal
        @test orthogonality_error(factor.U) < 1e-10
        @test orthogonality_error(factor.V) < 1e-10
        @test factor.S == randomized_svd(A, rank; power_iterations=2, rng=MersenneTwister(17)).S
        for B in (zeros(T, rows, columns), randn(rng, T, rows, 3) * randn(rng, T, 3, columns))
            F = randomized_svd(B, rank; rng=MersenneTwister(19))
            @test reconstruction_error(B, F) < 1e-10
            @test orthogonality_error(F.U) < 1e-10
            @test orthogonality_error(F.V) < 1e-10
        end
        full = randomized_svd(A, min(rows, columns); oversampling=0, rng=MersenneTwister(23))
        @test reconstruction_error(A, full) < 1e-10
    end
    A = randn(rng, 30, 20)
    @test_throws ArgumentError randomized_svd(A, 0)
    @test_throws ArgumentError randomized_svd(A, 21)
    @test_throws ArgumentError randomized_svd(A, 5; oversampling=-1)
    @test_throws ArgumentError randomized_svd(A, 5; power_iterations=-1)
    @test_throws DimensionMismatch randomized_svd!(RandomizedWorkspace(A, 5), zeros(20, 20), 5)
    for T in (Float64, ComplexF64)
        B = randn(rng, T, 128, 96)
        workspace = RandomizedWorkspace(B, 12)
        SVDScratch.randomized_range!(workspace, B, 1, rng)
        @test (@allocated SVDScratch.randomized_range!(workspace, B, 1, rng)) == 0
    end
end

@testset "Jacobi SVD" begin
    rng = MersenneTwister(17)
    for A in (
        randn(rng, 12, 8),
        randn(rng, 8, 12),
        randn(rng, ComplexF64, 10, 6),
        randn(rng, ComplexF64, 6, 10),
        randn(rng, 12, 3) * randn(rng, 3, 8),
        zeros(8, 5),
    )
        F = jacobi_svd(A)
        @test F.converged
        @test issorted(F.S; rev=true)
        @test reconstruction_error(A, F) < 1e-10
        @test orthogonality_error(F.U) < 1e-9
        @test orthogonality_error(F.V) < 1e-9
        @test norm(F.S - svdvals(A)) / max(norm(svdvals(A)), eps()) < 1e-10
    end

    A = randn(rng, ComplexF64, 10, 6)
    F = jacobi_svd!(copy(A))
    @test F.converged
    @test reconstruction_error(A, F) < 1e-10
    repeated = Matrix(Diagonal([9.0, 9.0, 3.0, 3.0, 1.0]))
    @test reconstruction_error(repeated, jacobi_svd(repeated)) < 1e-12
end

@testset "MIMO eigenchannels" begin
    rng = MersenneTwister(83)
    for n in (2, 4, 8)
        H = Matrix{ComplexF64}(undef, n, n)
        rayleigh_channel!(H, rng)
        @test eigenchannel_error(H, n) < 1e-10
        @test 1 <= select_streams(svd_scratch(H).S; relative_threshold=0.1) <= n
        diagonal = Matrix(Diagonal(ComplexF64.(n:-1:1) .* cis.(range(0.0, 1.0; length=n))))
        F = svd_scratch(diagonal)
        @test reconstruction_error(diagonal, F) < 1e-12
        @test isapprox(F.S, Float64.(n:-1:1); rtol=1e-12)
        @test 0.0 <= spatial_multiplexing_ber(n, n, n, 10.0; blocks=3, symbols_per_block=50, rng=MersenneTwister(89 + n)) <= 1.0
        @test mimo_capacity(n, n, 10.0; blocks=10, rng=MersenneTwister(97 + n)) > 0.0
    end
    adaptive = adaptive_multiplexing_ber(4, 4, 10.0; blocks=4, symbols_per_block=100, rng=MersenneTwister(111))
    @test 0.0 <= adaptive.ber <= 1.0
    @test 1.0 <= adaptive.average_streams <= 4.0
end

@testset "Golub-Reinsch SVD" begin
    rng = MersenneTwister(43)
    for A in (
        randn(rng, 8, 8),
        randn(rng, 14, 6),
        randn(rng, 6, 14),
        randn(rng, 12, 3) * randn(rng, 3, 7),
        Matrix(qr(randn(rng, 6, 6)).Q) * Diagonal([1.0, 1e-3, 1e-6, 1e-9, 1e-12, 1e-14]) * Matrix(qr(randn(rng, 6, 6)).Q)',
        zeros(9, 4),
    )
        F = golub_reinsch_svd(A)
        @test F.converged
        @test issorted(F.S; rev=true)
        @test frobenius(A - F.U * Diagonal(F.S) * transpose(F.V)) / max(frobenius(A), 1.0) < 1e-10
        @test orthogonality_error(F.U) < 1e-10
        @test orthogonality_error(F.V) < 1e-10
        @test norm(F.S - svdvals(A)) / max(norm(svdvals(A)), eps()) < 1e-10
    end

    @test isa(svd_scratch(randn(rng, 8, 5)), GolubReinschSVD)
    @test isa(svd_scratch(randn(rng, ComplexF64, 8, 5)), GolubReinschSVD)
    @test isa(svd_scratch(randn(rng, ComplexF64, 4, 4)), JacobiSVD)
    @test isa(svd_scratch(randn(rng, 8, 5); algorithm=:jacobi), JacobiSVD)
end

@testset "Complex Golub-Reinsch SVD" begin
    rng = MersenneTwister(61)
    for H in (
        randn(rng, ComplexF64, 8, 8),
        randn(rng, ComplexF64, 16, 6),
        randn(rng, ComplexF64, 6, 16),
        randn(rng, ComplexF64, 12, 3) * randn(rng, ComplexF64, 3, 7),
        zeros(ComplexF64, 9, 4),
    )
        F = golub_reinsch_svd(H)
        @test F.converged
        @test issorted(F.S; rev=true)
        @test reconstruction_error(H, F) < 1e-10
        @test orthogonality_error(F.U) < 1e-10
        @test orthogonality_error(F.V) < 1e-10
        @test norm(F.S - svdvals(H)) / max(norm(svdvals(H)), eps()) < 1e-10
    end
end

@testset "Householder bidiagonalization" begin
    rng = MersenneTwister(29)
    for A in (
        randn(rng, 10, 6),
        randn(rng, 17, 5),
        randn(rng, 8, 8),
        randn(rng, 12, 3) * randn(rng, 3, 7),
        zeros(9, 4),
    )
        F = bidiagonalize(A)
        @test bidiagonalization_error(A, F) < 1e-12
        @test orthogonality_error(F.U) < 1e-12
        @test orthogonality_error(F.V) < 1e-12
        @test all(iszero, tril(F.B, -1))
        @test all(iszero, triu(F.B, 2))
    end
end

@testset "PPM image I/O" begin
    source = tempname()
    output = tempname()
    payload = UInt8[0x20, 0x01, 0x02, 0x03, 0x04, 0x05]
    write(source, vcat(codeunits("P6\n2 1\n255\n"), payload))
    channels = read_ppm(source)
    @test channels[1][1, 1] == 0x20 / 255
    @test channels[3][1, 2] == 0x05 / 255
    write_ppm(output, channels)
    @test read(output) == read(source)
    for header in ("P6\r\n2 1\r\n255\r\n", "P6\n# real RGB\n2 1\n255\n")
        write(source, vcat(codeunits(header), payload))
        @test read_ppm(source) == channels
    end
    for invalid in ("", "P6\n", "P6\n-1 1\n255\n", "P6\n999999999999999999 99\n255\n", "P6\n2 1\n255\n")
        write(source, invalid)
        @test_throws ArgumentError read_ppm(source)
    end
    images = [rand(5, 6) for _ in 1:3]
    multiple = compress_image(images, (1, 3))
    @test all(isapprox(multiple[i][c], compress_image(images, rank)[c]; atol=1e-12) for (i, rank) in enumerate((1, 3)) for c in 1:3)
    @test_throws ArgumentError compress_image(Matrix{Float64}[], 1)
    @test_throws DimensionMismatch compress_image([zeros(2, 2), zeros(3, 2), zeros(2, 2)], 1)
    @test_throws DimensionMismatch image_metrics(images, [zeros(3, 4) for _ in 1:3], 1)
    rm(source)
    rm(output)
end
