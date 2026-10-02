export BidiagonalSVD, GolubReinschSVD, bidiagonal_svd, golub_reinsch_svd

struct BidiagonalSVD
    U::Matrix{Float64}
    S::Vector{Float64}
    V::Matrix{Float64}
    iterations::Int
    converged::Bool
end

struct GolubReinschSVD{T<:Number}
    U::Matrix{T}
    S::Vector{Float64}
    V::Matrix{T}
    iterations::Int
    converged::Bool
end

function rotate_columns!(Q::Matrix{Float64}, p::Int, q::Int, c::Float64, s::Float64)
    @inbounds for i in axes(Q, 1)
        first = Q[i, p]
        second = Q[i, q]
        Q[i, p] = c * first + s * second
        Q[i, q] = c * second - s * first
    end
    nothing
end

function bidiagonal_svd(B::AbstractMatrix{Float64}; max_iterations::Integer=128)
    rows, columns = size(B)
    rows == columns || throw(ArgumentError("B must be square"))
    columns > 0 || throw(ArgumentError("B must be nonempty"))
    max_iterations > 0 || throw(ArgumentError("max_iterations must be positive"))
    @inbounds for j in 1:columns
        for i in 1:rows
            abs(i - j) > 1 && !iszero(B[i, j]) && throw(ArgumentError("B must be bidiagonal"))
            i > j && !iszero(B[i, j]) && throw(ArgumentError("B must be upper bidiagonal"))
        end
    end
    diagonal = Vector{Float64}(undef, columns)
    superdiagonal = zeros(Float64, columns)
    input_scale = maximum(abs, B)
    isfinite(input_scale) || throw(ArgumentError("matrix entries must be finite"))
    input_scale = iszero(input_scale) ? 1.0 : input_scale
    @inbounds for j in 1:columns
        diagonal[j] = B[j, j] / input_scale
        if j > 1
            superdiagonal[j] = B[j - 1, j] / input_scale
        end
    end
    U = Matrix{Float64}(I, columns, columns)
    V = Matrix{Float64}(I, columns, columns)
    scale = sum(abs, diagonal) + sum(abs, superdiagonal)
    threshold = eps(Float64) * scale
    iterations = 0
    for k in columns:-1:1
        converged_k = false
        for _ in 1:max_iterations
            iterations += 1
            flag = true
            l = k
            @inbounds for candidate in k:-1:1
                l = candidate
                if candidate == 1 || abs(superdiagonal[candidate]) <= threshold
                    flag = false
                    break
                end
                abs(diagonal[candidate - 1]) <= threshold && break
            end
            if flag
                nm = l - 1
                c = 0.0
                s = 1.0
                @inbounds for i in l:k
                    f = s * superdiagonal[i]
                    superdiagonal[i] = c * superdiagonal[i]
                    abs(f) <= threshold && break
                    g = diagonal[i]
                    h = hypot(f, g)
                    diagonal[i] = h
                    c = g / h
                    s = -f / h
                    rotate_columns!(U, nm, i, c, s)
                end
            end
            z = diagonal[k]
            if l == k
                if z < 0.0
                    diagonal[k] = -z
                    @inbounds for i in 1:columns
                        V[i, k] = -V[i, k]
                    end
                end
                converged_k = true
                break
            end
            nm = k - 1
            x = diagonal[l]
            y = diagonal[nm]
            g = superdiagonal[nm]
            h = superdiagonal[k]
            shift = h == 0.0 || y == 0.0 ? 0.0 : ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y)
            shift_root = hypot(shift, 1.0)
            denominator = shift + copysign(shift_root, shift)
            f = x == 0.0 || denominator == 0.0 ? 0.0 : ((x - z) * (x + z) + h * (y / denominator - h)) / x
            c = 1.0
            s = 1.0
            @inbounds for j in l:nm
                i = j + 1
                g = superdiagonal[i]
                y = diagonal[i]
                h = s * g
                g = c * g
                z = hypot(f, h)
                superdiagonal[j] = z
                c = z == 0.0 ? 1.0 : f / z
                s = z == 0.0 ? 0.0 : h / z
                f = x * c + g * s
                g = g * c - x * s
                h = y * s
                y *= c
                rotate_columns!(V, j, i, c, s)
                z = hypot(f, h)
                diagonal[j] = z
                c = z == 0.0 ? 1.0 : f / z
                s = z == 0.0 ? 0.0 : h / z
                f = c * g + s * y
                x = c * y - s * g
                rotate_columns!(U, j, i, c, s)
            end
            superdiagonal[l] = 0.0
            superdiagonal[k] = f
            diagonal[k] = x
        end
        if !converged_k
            diagonal .*= input_scale
            return BidiagonalSVD(U, diagonal, V, iterations, false)
        end
    end
    order = sortperm(diagonal; rev=true)
    diagonal .*= input_scale
    BidiagonalSVD(U[:, order], diagonal[order], V[:, order], iterations, true)
end

function golub_reinsch_svd(A::AbstractMatrix{Float64}; max_iterations::Integer=128)
    rows, columns = size(A)
    rows > 0 || throw(ArgumentError("A must have at least one row"))
    columns > 0 || throw(ArgumentError("A must have at least one column"))
    if rows < columns
        factorization = golub_reinsch_svd(Matrix(transpose(A)); max_iterations)
        return GolubReinschSVD(factorization.V, factorization.S, factorization.U, factorization.iterations, factorization.converged)
    end
    reduction = bidiagonalize(A)
    diagonalization = bidiagonal_svd(reduction.B; max_iterations)
    GolubReinschSVD(reduction.U * diagonalization.U, diagonalization.S, reduction.V * diagonalization.V, diagonalization.iterations, diagonalization.converged)
end

function reconstruction_error(A::AbstractMatrix, F::GolubReinschSVD)
    denominator = frobenius(A)
    residual = frobenius(A - F.U * Diagonal(F.S) * adjoint(F.V))
    denominator == 0.0 ? residual : residual / denominator
end
