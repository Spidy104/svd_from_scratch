export Bidiagonalization, bidiagonalize, bidiagonalization_error

struct Bidiagonalization
    U::Matrix{Float64}
    B::Matrix{Float64}
    V::Matrix{Float64}
end

function left_reflector!(A::Matrix{Float64}, row::Int, column::Int)
    rows = size(A, 1)
    alpha = A[row, column]
    tail_norm2 = 0.0
    @inbounds for i in (row + 1):rows
        tail_norm2 += A[i, column] * A[i, column]
    end
    normx = hypot(alpha, sqrt(tail_norm2))
    normx == 0.0 && return 0.0
    beta = -copysign(normx, alpha == 0.0 ? 1.0 : alpha)
    denominator = alpha - beta
    @inbounds for i in (row + 1):rows
        A[i, column] /= denominator
    end
    A[row, column] = beta
    (beta - alpha) / beta
end

function right_reflector!(A::Matrix{Float64}, row::Int, column::Int)
    columns = size(A, 2)
    alpha = A[row, column]
    tail_norm2 = 0.0
    @inbounds for j in (column + 1):columns
        tail_norm2 += A[row, j] * A[row, j]
    end
    normx = hypot(alpha, sqrt(tail_norm2))
    normx == 0.0 && return 0.0
    beta = -copysign(normx, alpha == 0.0 ? 1.0 : alpha)
    denominator = alpha - beta
    @inbounds for j in (column + 1):columns
        A[row, j] /= denominator
    end
    A[row, column] = beta
    (beta - alpha) / beta
end

function apply_left_reflector!(A::Matrix{Float64}, storage::Matrix{Float64}, tau::Float64, row::Int, vector_column::Int, first_column::Int)
    tau == 0.0 && return nothing
    rows, columns = size(A)
    @inbounds for j in first_column:columns
        projection = A[row, j]
        for i in (row + 1):rows
            projection += storage[i, vector_column] * A[i, j]
        end
        projection *= tau
        A[row, j] -= projection
        for i in (row + 1):rows
            A[i, j] -= storage[i, vector_column] * projection
        end
    end
    nothing
end

function apply_right_reflector!(A::Matrix{Float64}, storage::Matrix{Float64}, tau::Float64, row::Int, vector_row::Int, column::Int, first_row::Int)
    tau == 0.0 && return nothing
    rows, columns = size(A)
    @inbounds for i in first_row:rows
        projection = A[i, column]
        for j in (column + 1):columns
            projection += A[i, j] * storage[vector_row, j]
        end
        projection *= tau
        A[i, column] -= projection
        for j in (column + 1):columns
            A[i, j] -= projection * storage[vector_row, j]
        end
    end
    nothing
end

function left_factor(storage::Matrix{Float64}, taus::Vector{Float64})
    rows, columns = size(storage)
    U = zeros(Float64, rows, columns)
    @inbounds for j in 1:columns
        U[j, j] = 1.0
    end
    for row in columns:-1:1
        apply_left_reflector!(U, storage, taus[row], row, row, 1)
    end
    U
end

function right_factor(storage::Matrix{Float64}, taus::Vector{Float64})
    columns = size(storage, 2)
    V = Matrix{Float64}(I, columns, columns)
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
                V[j, k] -= storage[row, j] * projection
            end
        end
    end
    V
end

function bidiagonalize(A::AbstractMatrix{Float64})
    rows, columns = size(A)
    rows >= columns || throw(ArgumentError("bidiagonalize requires a tall or square matrix"))
    rows > 0 || throw(ArgumentError("A must have at least one row"))
    columns > 0 || throw(ArgumentError("A must have at least one column"))
    storage = Matrix{Float64}(A)
    input_scale = normalize_matrix!(storage)
    left_taus = zeros(Float64, columns)
    right_taus = zeros(Float64, max(columns - 1, 0))
    @inbounds for j in 1:columns
        left_taus[j] = left_reflector!(storage, j, j)
        apply_left_reflector!(storage, storage, left_taus[j], j, j, j + 1)
        if j < columns
            right_taus[j] = right_reflector!(storage, j, j + 1)
            apply_right_reflector!(storage, storage, right_taus[j], j, j, j + 1, j + 1)
        end
    end
    U = left_factor(storage, left_taus)
    V = right_factor(storage, right_taus)
    B = zeros(Float64, columns, columns)
    @inbounds for j in 1:columns
        B[j, j] = storage[j, j] * input_scale
        if j < columns
            B[j, j + 1] = storage[j, j + 1] * input_scale
        end
    end
    Bidiagonalization(U, B, V)
end

function bidiagonalization_error(A::AbstractMatrix{Float64}, F::Bidiagonalization)
    denominator = frobenius(A)
    residual = frobenius(A - F.U * F.B * transpose(F.V))
    denominator == 0.0 ? residual : residual / denominator
end
