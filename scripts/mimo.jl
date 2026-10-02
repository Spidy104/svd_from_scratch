using Printf
using Random
using SVDScratch

rng = MersenneTwister(71)
for n in (2, 4, 8)
    H = Matrix{ComplexF64}(undef, n, n)
    rayleigh_channel!(H, rng)
    println("$(n)x$(n) eigenchannel error: $(eigenchannel_error(H, n))")
    println("$(n)x$(n) active streams at 10% threshold: $(select_streams(svd_scratch(H).S))")
end
println()
println("SNR dB | 2x2 BER | 4x4 BER | 8x8 BER | 2x2 C | 4x4 C | 8x8 C")
println("-----------------------------------------------------------------")
for snr_db in (-5, 5, 15)
    ber2 = spatial_multiplexing_ber(2, 2, 2, snr_db; blocks=100, symbols_per_block=1_000, rng=MersenneTwister(1_000 + snr_db))
    ber4 = spatial_multiplexing_ber(4, 4, 4, snr_db; blocks=100, symbols_per_block=1_000, rng=MersenneTwister(2_000 + snr_db))
    ber8 = spatial_multiplexing_ber(8, 8, 8, snr_db; blocks=100, symbols_per_block=1_000, rng=MersenneTwister(3_000 + snr_db))
    cap2 = mimo_capacity(2, 2, snr_db; blocks=400, rng=MersenneTwister(4_000 + snr_db))
    cap4 = mimo_capacity(4, 4, snr_db; blocks=400, rng=MersenneTwister(5_000 + snr_db))
    cap8 = mimo_capacity(8, 8, snr_db; blocks=400, rng=MersenneTwister(6_000 + snr_db))
    @printf("%6d | %.5f | %.5f | %.5f | %.3f | %.3f | %.3f\n", snr_db, ber2, ber4, ber8, cap2, cap4, cap8)
end

adaptive = adaptive_multiplexing_ber(4, 4, 5.0; relative_threshold=0.35, blocks=120, symbols_per_block=1_000, rng=MersenneTwister(7_000))
println()
println("adaptive rank at 5 dB: $(adaptive.average_streams) average streams, BER $(adaptive.ber)")
