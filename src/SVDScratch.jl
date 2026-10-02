module SVDScratch

using LinearAlgebra
using Random

export JacobiSVD, jacobi_svd, jacobi_svd!, svd_scratch, reconstruction_error, orthogonality_error, frobenius

include("Bidiagonalization.jl")
include("BidiagonalQR.jl")
include("ComplexGolubReinsch.jl")
include("MIMO.jl")
include("ImageCompression.jl")
include("RandomizedSVD.jl")

function svd_scratch(A::AbstractMatrix{Float64}; algorithm::Symbol=:auto, kwargs...)
    algorithm === :auto && return golub_reinsch_svd(A; kwargs...)
    algorithm === :golub_reinsch && return golub_reinsch_svd(A; kwargs...)
    algorithm === :jacobi && return jacobi_svd(A; kwargs...)
    throw(ArgumentError("algorithm must be :auto, :golub_reinsch, or :jacobi"))
end

function svd_scratch(A::AbstractMatrix{ComplexF64}; algorithm::Symbol=:auto, kwargs...)
    algorithm === :auto && return min(size(A)...) <= 4 ? jacobi_svd(A; kwargs...) : golub_reinsch_svd(A; kwargs...)
    algorithm === :jacobi && return jacobi_svd(A; kwargs...)
    algorithm === :golub_reinsch && return golub_reinsch_svd(A; kwargs...)
    throw(ArgumentError("algorithm must be :auto, :golub_reinsch, or :jacobi"))
end

struct JacobiSVD{T<:Number}
    U::Matrix{T}
    S::Vector{Float64}
    V::Matrix{T}
    sweeps::Int
    converged::Bool
end

function frobenius(A::AbstractMatrix)
    norm(A)
end

function normalize_matrix!(A::Matrix)
    scale = maximum(abs, A)
    isfinite(scale) || throw(ArgumentError("matrix entries must be finite"))
    scale = iszero(scale) ? 1.0 : scale
    @inbounds for i in eachindex(A)
        A[i] /= scale
    end
    scale
end

@inline function column_norm2(A::AbstractMatrix, j::Int)
    total = 0.0
    @inbounds for i in axes(A, 1)
        total += abs2(A[i, j])
    end
    total
end

@inline function column_inner(A::AbstractMatrix, p::Int, q::Int)
    total = zero(eltype(A))
    @inbounds for i in axes(A, 1)
        total += conj(A[i, p]) * A[i, q]
    end
    total
end

@inline function jacobi_t(tau::Float64)
    root = sqrt(1.0 + tau * tau)
    tau >= 0.0 ? inv(tau + root) : -inv(root - tau)
end

function rotate_columns!(B::AbstractMatrix{T}, V::AbstractMatrix{T}, p::Int, q::Int, c::Float64, s::Float64, phase::T) where {T<:Union{Float64,ComplexF64}}
    phase_conjugate = conj(phase)
    @inbounds for i in axes(B, 1)
        bp = B[i, p]
        bq = B[i, q]
        B[i, p] = c * bp - s * phase_conjugate * bq
        B[i, q] = s * phase * bp + c * bq
    end
    @inbounds for i in axes(V, 1)
        vp = V[i, p]
        vq = V[i, q]
        V[i, p] = c * vp - s * phase_conjugate * vq
        V[i, q] = s * phase * vp + c * vq
    end
    nothing
end

function complete_column!(U::AbstractMatrix{T}, j::Int) where {T<:Union{Float64,ComplexF64}}
    rows = size(U, 1)
    @inbounds for candidate in 1:rows
        for i in 1:rows
            U[i, j] = zero(T)
        end
        U[candidate, j] = one(T)
        for _ in 1:2
            for p in 1:(j - 1)
                coefficient = column_inner(U, p, j)
                for i in 1:rows
                    U[i, j] -= U[i, p] * coefficient
                end
            end
        end
        squared_norm = column_norm2(U, j)
        if squared_norm > eps(Float64)
            scale = inv(sqrt(squared_norm))
            for i in 1:rows
                U[i, j] *= scale
            end
            return nothing
        end
    end
    throw(ArgumentError("unable to complete an orthonormal basis"))
end

function finalize_svd(B::Matrix{T}, V::Matrix{T}, sweeps::Int, converged::Bool, rtol::Float64) where {T<:Union{Float64,ComplexF64}}
    rows, columns = size(B)
    singular_unsorted = Vector{Float64}(undef, columns)
    @inbounds for j in 1:columns
        singular_unsorted[j] = sqrt(column_norm2(B, j))
    end
    order = sortperm(singular_unsorted; rev=true)
    S = singular_unsorted[order]
    U = Matrix{T}(undef, rows, columns)
    V_sorted = Matrix{T}(undef, columns, columns)
    cutoff = isempty(S) ? 0.0 : S[1] * max(rtol, eps(Float64))
    @inbounds for j in 1:columns
        source = order[j]
        for i in 1:columns
            V_sorted[i, j] = V[i, source]
        end
        if S[j] > cutoff
            scale = inv(S[j])
            for i in 1:rows
                U[i, j] = B[i, source] * scale
            end
            for _ in 1:2
                for p in 1:(j - 1)
                    coefficient = column_inner(U, p, j)
                    for i in 1:rows
                        U[i, j] -= U[i, p] * coefficient
                    end
                end
            end
            squared_norm = column_norm2(U, j)
            if squared_norm > 0.25
                scale = inv(sqrt(squared_norm))
                for i in 1:rows
                    U[i, j] *= scale
                end
            else
                complete_column!(U, j)
            end
        else
            complete_column!(U, j)
        end
    end
    JacobiSVD(U, S, V_sorted, sweeps, converged)
end

function validate_parameters(rows::Int, columns::Int, rtol::Real, max_sweeps::Integer)
    rows > 0 || throw(ArgumentError("A must have at least one row"))
    columns > 0 || throw(ArgumentError("A must have at least one column"))
    rtol > 0.0 || throw(ArgumentError("rtol must be positive"))
    max_sweeps > 0 || throw(ArgumentError("max_sweeps must be positive"))
    nothing
end

function jacobi_svd!(B::Matrix{T}; rtol::Real=64eps(Float64), max_sweeps::Integer=100) where {T<:Union{Float64,ComplexF64}}
    rows, columns = size(B)
    validate_parameters(rows, columns, rtol, max_sweeps)
    rows >= columns || throw(ArgumentError("jacobi_svd! requires a tall or square matrix"))
    input_scale = normalize_matrix!(B)
    tolerance = Float64(rtol)
    V = Matrix{T}(I, columns, columns)
    norms = Vector{Float64}(undef, columns)
    @inbounds for j in 1:columns
        norms[j] = column_norm2(B, j)
    end
    for sweep in 1:max_sweeps
        rotated = false
        @inbounds for q in 2:columns
            for p in 1:(q - 1)
                alpha = norms[p]
                beta = norms[q]
                gamma = column_inner(B, p, q)
                magnitude = abs(gamma)
                if magnitude > tolerance * sqrt(alpha * beta)
                    tau = (beta - alpha) / (2.0 * magnitude)
                    t = jacobi_t(tau)
                    c = inv(sqrt(1.0 + t * t))
                    s = c * t
                    phase = gamma / magnitude
                    rotate_columns!(B, V, p, q, c, s, phase)
                    cross = 2.0 * c * s * magnitude
                    norms[p] = max(0.0, c * c * alpha - cross + s * s * beta)
                    norms[q] = max(0.0, s * s * alpha + cross + c * c * beta)
                    rotated = true
                end
            end
        end
        if !rotated
            factor = finalize_svd(B, V, sweep, true, tolerance)
            factor.S .*= input_scale
            return factor
        end
        @inbounds for j in 1:columns
            norms[j] = column_norm2(B, j)
        end
    end
    factor = finalize_svd(B, V, Int(max_sweeps), false, tolerance)
    factor.S .*= input_scale
    factor
end

function jacobi_svd(A::AbstractMatrix{T}; rtol::Real=64eps(Float64), max_sweeps::Integer=100) where {T<:Union{Float64,ComplexF64}}
    rows, columns = size(A)
    validate_parameters(rows, columns, rtol, max_sweeps)
    if rows < columns
        factorization = jacobi_svd!(Matrix(adjoint(A)); rtol, max_sweeps)
        return JacobiSVD(factorization.V, factorization.S, factorization.U, factorization.sweeps, factorization.converged)
    end
    jacobi_svd!(Matrix{T}(A); rtol, max_sweeps)
end

function reconstruction_error(A::AbstractMatrix, F::JacobiSVD)
    denominator = frobenius(A)
    residual = frobenius(A - F.U * Diagonal(F.S) * adjoint(F.V))
    denominator == 0.0 ? residual : residual / denominator
end

function orthogonality_error(Q::AbstractMatrix)
    columns = size(Q, 2)
    frobenius(adjoint(Q) * Q - Matrix{eltype(Q)}(I, columns, columns))
end

end
