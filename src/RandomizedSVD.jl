export RandomizedSVD, RandomizedWorkspace, randomized_svd, randomized_svd!

struct RandomizedSVD{T<:Number}
    U::Matrix{T}
    S::Vector{Float64}
    V::Matrix{T}
    converged::Bool
end

struct RandomizedWorkspace{T<:Union{Float64,ComplexF64}}
    Q::Matrix{T}
    Z::Matrix{T}
    B::Matrix{T}
end

function RandomizedWorkspace(A::AbstractMatrix{T}, rank::Integer; oversampling::Integer=8) where {T<:Union{Float64,ComplexF64}}
    rows, columns = size(A)
    1 <= rank <= min(rows, columns) || throw(ArgumentError("rank must be in 1:min(size(A)...)"))
    oversampling >= 0 || throw(ArgumentError("oversampling must be nonnegative"))
    width = rank + min(oversampling, min(rows, columns) - rank)
    RandomizedWorkspace(Matrix{T}(undef, rows, width), Matrix{T}(undef, columns, width), Matrix{T}(undef, width, columns))
end

function orthonormalize!(Q::Matrix)
    @inbounds for j in axes(Q, 2)
        original_norm = sqrt(column_norm2(Q, j))
        for _ in 1:2
            for p in 1:(j - 1)
                projection = column_inner(Q, p, j)
                for i in axes(Q, 1)
                    Q[i, j] -= Q[i, p] * projection
                end
            end
        end
        magnitude = sqrt(column_norm2(Q, j))
        if magnitude <= 64eps(Float64) * original_norm
            complete_column!(Q, j)
        else
            for i in axes(Q, 1)
                Q[i, j] /= magnitude
            end
        end
    end
    Q
end

function randomized_range!(workspace::RandomizedWorkspace, A::AbstractMatrix, power_iterations::Integer, rng::AbstractRNG, scale::Float64=1.0)
    Q, Z, B = workspace.Q, workspace.Z, workspace.B
    randn!(rng, Z)
    mul!(Q, A, Z, inv(scale), 0.0)
    orthonormalize!(Q)
    for _ in 1:power_iterations
        mul!(Z, adjoint(A), Q, inv(scale), 0.0)
        orthonormalize!(Z)
        mul!(Q, A, Z, inv(scale), 0.0)
        orthonormalize!(Q)
    end
    mul!(B, adjoint(Q), A, inv(scale), 0.0)
    nothing
end

function randomized_svd!(workspace::RandomizedWorkspace{T}, A::AbstractMatrix{T}, rank::Integer; power_iterations::Integer=1, rng::AbstractRNG=Random.default_rng()) where {T}
    size(A) == (size(workspace.Q, 1), size(workspace.Z, 1)) || throw(DimensionMismatch("workspace does not match A"))
    width = size(workspace.Q, 2)
    size(workspace.Z, 2) == width && size(workspace.B) == (width, size(A, 2)) && width <= min(size(A)...) || throw(DimensionMismatch("invalid workspace buffers"))
    1 <= rank <= size(workspace.Q, 2) || throw(ArgumentError("rank exceeds workspace width"))
    power_iterations >= 0 || throw(ArgumentError("power_iterations must be nonnegative"))
    scale = maximum(abs, A)
    isfinite(scale) || throw(ArgumentError("matrix entries must be finite"))
    scale = iszero(scale) || 1e-100 <= scale <= 1e100 ? 1.0 : max(scale, floatmin(Float64))
    randomized_range!(workspace, A, power_iterations, rng, scale)
    factor = golub_reinsch_svd(workspace.B)
    U = Matrix{T}(undef, size(A, 1), rank)
    mul!(U, workspace.Q, view(factor.U, :, 1:rank))
    factor.S .*= scale
    RandomizedSVD(U, factor.S[1:rank], factor.V[:, 1:rank], factor.converged)
end

function randomized_svd(A::AbstractMatrix{T}, rank::Integer; oversampling::Integer=8, power_iterations::Integer=1, rng::AbstractRNG=Random.default_rng()) where {T<:Union{Float64,ComplexF64}}
    workspace = RandomizedWorkspace(A, rank; oversampling)
    randomized_svd!(workspace, A, rank; power_iterations, rng)
end

function reconstruction_error(A::AbstractMatrix, factor::RandomizedSVD)
    residual = frobenius(A - factor.U * Diagonal(factor.S) * adjoint(factor.V))
    residual / max(frobenius(A), eps(Float64))
end
