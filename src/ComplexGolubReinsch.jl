struct ComplexBidiagonalization
    U::Matrix{ComplexF64}
    B::Matrix{ComplexF64}
    V::Matrix{ComplexF64}
end

@inline function unit_phase(z::ComplexF64)
    iszero(z) ? one(ComplexF64) : z / abs(z)
end

function left_reflector_complex!(A::Matrix{ComplexF64}, row::Int, column::Int)
    rows = size(A, 1)
    alpha = A[row, column]
    tail_norm2 = 0.0
    @inbounds for i in (row + 1):rows
        tail_norm2 += abs2(A[i, column])
    end
    normx = hypot(abs(alpha), sqrt(tail_norm2))
    normx == 0.0 && return 0.0
    beta = -unit_phase(alpha) * normx
    denominator = alpha - beta
    @inbounds for i in (row + 1):rows
        A[i, column] /= denominator
    end
    A[row, column] = beta
    real((beta - alpha) / beta)
end

function right_reflector_complex!(A::Matrix{ComplexF64}, row::Int, column::Int)
    columns = size(A, 2)
    alpha = A[row, column]
    tail_norm2 = 0.0
    @inbounds for j in (column + 1):columns
        tail_norm2 += abs2(A[row, j])
    end
    normx = hypot(abs(alpha), sqrt(tail_norm2))
    normx == 0.0 && return 0.0
    beta = -unit_phase(alpha) * normx
    denominator = alpha - beta
    @inbounds for j in (column + 1):columns
        A[row, j] /= denominator
    end
    A[row, column] = beta
    real((beta - alpha) / beta)
end

function apply_left_reflector_complex!(A::Matrix{ComplexF64}, storage::Matrix{ComplexF64}, tau::Float64, row::Int, vector_column::Int, first_column::Int)
    tau == 0.0 && return nothing
    rows, columns = size(A)
    @inbounds for j in first_column:columns
        projection = A[row, j]
        for i in (row + 1):rows
            projection += conj(storage[i, vector_column]) * A[i, j]
        end
        projection *= tau
        A[row, j] -= projection
        for i in (row + 1):rows
            A[i, j] -= storage[i, vector_column] * projection
        end
    end
    nothing
end

function apply_right_reflector_complex!(A::Matrix{ComplexF64}, storage::Matrix{ComplexF64}, tau::Float64, row::Int, vector_row::Int, column::Int, first_row::Int)
    tau == 0.0 && return nothing
    rows, columns = size(A)
    @inbounds for i in first_row:rows
        projection = A[i, column]
        for j in (column + 1):columns
            projection += A[i, j] * conj(storage[vector_row, j])
        end
        projection *= tau
        A[i, column] -= projection
        for j in (column + 1):columns
            A[i, j] -= projection * storage[vector_row, j]
        end
    end
    nothing
end

function complex_left_factor(storage::Matrix{ComplexF64}, taus::Vector{Float64})
    rows, columns = size(storage)
    U = zeros(ComplexF64, rows, columns)
    @inbounds for j in 1:columns
        U[j, j] = 1.0
    end
    for row in columns:-1:1
        apply_left_reflector_complex!(U, storage, taus[row], row, row, 1)
    end
    U
end

function complex_right_factor(storage::Matrix{ComplexF64}, taus::Vector{Float64})
    columns = size(storage, 2)
    V = Matrix{ComplexF64}(I, columns, columns)
    for row in (columns - 1):-1:1
        tau = taus[row]
        tau == 0.0 && continue
        start = row + 1
        @inbounds for k in 1:columns
            projection = V[start, k]
            for j in (start + 1):columns
                projection += storage[row, j] * V[j, k]
            end
            projection *= tau
            V[start, k] -= projection
            for j in (start + 1):columns
                V[j, k] -= conj(storage[row, j]) * projection
            end
        end
    end
    V
end

function complex_bidiagonalize(A::AbstractMatrix{ComplexF64})
    rows, columns = size(A)
    rows >= columns || throw(ArgumentError("complex_bidiagonalize requires a tall or square matrix"))
    rows > 0 || throw(ArgumentError("A must have at least one row"))
    columns > 0 || throw(ArgumentError("A must have at least one column"))
    storage = Matrix{ComplexF64}(A)
    input_scale = normalize_matrix!(storage)
    left_taus = zeros(Float64, columns)
    right_taus = zeros(Float64, max(columns - 1, 0))
    @inbounds for j in 1:columns
        left_taus[j] = left_reflector_complex!(storage, j, j)
        apply_left_reflector_complex!(storage, storage, left_taus[j], j, j, j + 1)
        if j < columns
            right_taus[j] = right_reflector_complex!(storage, j, j + 1)
            apply_right_reflector_complex!(storage, storage, right_taus[j], j, j, j + 1, j + 1)
        end
    end
    U = complex_left_factor(storage, left_taus)
    V = complex_right_factor(storage, right_taus)
    B = zeros(ComplexF64, columns, columns)
    @inbounds for j in 1:columns
        B[j, j] = storage[j, j] * input_scale
        if j < columns
            B[j, j + 1] = storage[j, j + 1] * input_scale
        end
    end
    ComplexBidiagonalization(U, B, V)
end

function real_bidiagonal(B::Matrix{ComplexF64})
    columns = size(B, 1)
    real_B = zeros(Float64, columns, columns)
    left_phase = Vector{ComplexF64}(undef, columns)
    right_phase = Vector{ComplexF64}(undef, columns)
    right_phase[1] = 1.0
    @inbounds for j in 1:columns
        diagonal_value = B[j, j] * right_phase[j]
        left_phase[j] = unit_phase(diagonal_value)
        real_B[j, j] = abs(diagonal_value)
        if j < columns
            superdiagonal_value = conj(left_phase[j]) * B[j, j + 1]
            right_phase[j + 1] = conj(unit_phase(superdiagonal_value))
            real_B[j, j + 1] = abs(superdiagonal_value)
        end
    end
    real_B, left_phase, right_phase
end

function golub_reinsch_svd(A::AbstractMatrix{ComplexF64}; max_iterations::Integer=128)
    rows, columns = size(A)
    rows > 0 || throw(ArgumentError("A must have at least one row"))
    columns > 0 || throw(ArgumentError("A must have at least one column"))
    if rows < columns
        factorization = golub_reinsch_svd(Matrix(adjoint(A)); max_iterations)
        return GolubReinschSVD(factorization.V, factorization.S, factorization.U, factorization.iterations, factorization.converged)
    end
    reduction = complex_bidiagonalize(A)
    real_B, left_phase, right_phase = real_bidiagonal(reduction.B)
    diagonalization = bidiagonal_svd(real_B; max_iterations)
    U = reduction.U * (Diagonal(left_phase) * diagonalization.U)
    V = reduction.V * (Diagonal(right_phase) * diagonalization.V)
    GolubReinschSVD(U, diagonalization.S, V, diagonalization.iterations, diagonalization.converged)
end
