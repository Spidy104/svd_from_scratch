export rayleigh_channel!, eigenchannel_error, select_streams, spatial_multiplexing_ber, adaptive_multiplexing_ber, mimo_capacity

const INV_SQRT2 = inv(sqrt(2.0))

function rayleigh_channel!(H::Matrix{ComplexF64}, rng::AbstractRNG)
    @inbounds for i in eachindex(H)
        H[i] = ComplexF64(randn(rng) * INV_SQRT2, randn(rng) * INV_SQRT2)
    end
    H
end

function select_streams(singular::AbstractVector{<:Real}; relative_threshold::Real=0.1)
    isempty(singular) && return 0
    relative_threshold >= 0.0 || throw(ArgumentError("relative_threshold must be nonnegative"))
    threshold = singular[1] * relative_threshold
    count(value -> value > threshold, singular)
end

function eigenchannel_error(H::Matrix{ComplexF64}, streams::Integer; algorithm::Symbol=:auto)
    streams > 0 || throw(ArgumentError("streams must be positive"))
    streams <= min(size(H)...) || throw(ArgumentError("streams exceeds the MIMO rank limit"))
    F = svd_scratch(H; algorithm)
    U = @view F.U[:, 1:streams]
    V = @view F.V[:, 1:streams]
    equivalent = adjoint(U) * H * V
    for i in 1:streams
        equivalent[i, i] -= F.S[i]
    end
    frobenius(equivalent) / max(frobenius(H), 1.0)
end

@inline function qpsk_symbol_mimo(rng::AbstractRNG)
    re = rand(rng, Bool) ? INV_SQRT2 : -INV_SQRT2
    im = rand(rng, Bool) ? INV_SQRT2 : -INV_SQRT2
    ComplexF64(re, im)
end

@inline function qpsk_errors(a::ComplexF64, b::ComplexF64)
    ((real(a) >= 0.0) != (real(b) >= 0.0)) + ((imag(a) >= 0.0) != (imag(b) >= 0.0))
end

function qpsk_mode_errors!(symbols::Vector{ComplexF64}, received::Vector{ComplexF64}, singular::AbstractVector{<:Real}, snr::Float64, rng::AbstractRNG)
    streams = length(singular)
    gain = sqrt(snr / streams)
    errors = 0
    @inbounds for i in 1:streams
        symbol = qpsk_symbol_mimo(rng)
        received_symbol = gain * singular[i] * symbol + ComplexF64(randn(rng) * INV_SQRT2, randn(rng) * INV_SQRT2)
        symbols[i] = symbol
        received[i] = received_symbol
        errors += qpsk_errors(symbol, received_symbol)
    end
    errors
end

function spatial_multiplexing_ber(nt::Integer, nr::Integer, streams::Integer, snr_db::Real; blocks::Integer=200, symbols_per_block::Integer=2_000, algorithm::Symbol=:auto, rng::AbstractRNG=MersenneTwister(1))
    nt > 0 || throw(ArgumentError("nt must be positive"))
    nr > 0 || throw(ArgumentError("nr must be positive"))
    streams > 0 || throw(ArgumentError("streams must be positive"))
    streams <= min(nt, nr) || throw(ArgumentError("streams exceeds the MIMO rank limit"))
    blocks > 0 || throw(ArgumentError("blocks must be positive"))
    symbols_per_block > 0 || throw(ArgumentError("symbols_per_block must be positive"))
    H = Matrix{ComplexF64}(undef, nr, nt)
    symbols = Vector{ComplexF64}(undef, streams)
    received = similar(symbols)
    snr = 10.0^(Float64(snr_db) / 10.0)
    errors = 0
    @inbounds for _ in 1:blocks
        rayleigh_channel!(H, rng)
        F = svd_scratch(H; algorithm)
        singular = @view F.S[1:streams]
        for _ in 1:symbols_per_block
            errors += qpsk_mode_errors!(symbols, received, singular, snr, rng)
        end
    end
    errors / (2.0 * blocks * symbols_per_block * streams)
end

function adaptive_multiplexing_ber(nt::Integer, nr::Integer, snr_db::Real; relative_threshold::Real=0.1, blocks::Integer=200, symbols_per_block::Integer=2_000, algorithm::Symbol=:auto, rng::AbstractRNG=MersenneTwister(3))
    nt > 0 || throw(ArgumentError("nt must be positive"))
    nr > 0 || throw(ArgumentError("nr must be positive"))
    relative_threshold >= 0.0 || throw(ArgumentError("relative_threshold must be nonnegative"))
    blocks > 0 || throw(ArgumentError("blocks must be positive"))
    symbols_per_block > 0 || throw(ArgumentError("symbols_per_block must be positive"))
    maximum_streams = min(nt, nr)
    H = Matrix{ComplexF64}(undef, nr, nt)
    symbols = Vector{ComplexF64}(undef, maximum_streams)
    received = similar(symbols)
    snr = 10.0^(Float64(snr_db) / 10.0)
    errors = 0
    bits = 0
    streams_total = 0
    @inbounds for _ in 1:blocks
        rayleigh_channel!(H, rng)
        F = svd_scratch(H; algorithm)
        streams = max(select_streams(F.S; relative_threshold), 1)
        singular = @view F.S[1:streams]
        for _ in 1:symbols_per_block
            errors += qpsk_mode_errors!(symbols, received, singular, snr, rng)
        end
        bits += 2 * symbols_per_block * streams
        streams_total += streams
    end
    (ber=errors / bits, average_streams=streams_total / blocks)
end

function mimo_capacity(nt::Integer, nr::Integer, snr_db::Real; blocks::Integer=1_000, algorithm::Symbol=:auto, rng::AbstractRNG=MersenneTwister(2))
    nt > 0 || throw(ArgumentError("nt must be positive"))
    nr > 0 || throw(ArgumentError("nr must be positive"))
    blocks > 0 || throw(ArgumentError("blocks must be positive"))
    H = Matrix{ComplexF64}(undef, nr, nt)
    snr = 10.0^(Float64(snr_db) / 10.0)
    capacity = 0.0
    @inbounds for _ in 1:blocks
        rayleigh_channel!(H, rng)
        F = svd_scratch(H; algorithm)
        for value in F.S
            capacity += log2(1.0 + (snr / nt) * value * value)
        end
    end
    capacity / blocks
end
