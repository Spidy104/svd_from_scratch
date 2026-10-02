using LinearAlgebra
using Printf
using Random
using SVDScratch

BLAS.set_num_threads(1)
root = dirname(@__DIR__)
directory = mktempdir()
source = joinpath(root, "assets", "real_landscape.jpg")
ppm = joinpath(directory, "original.ppm")
run(`magick $source -resize 512x512^ -gravity center -extent 512x512 -depth 8 $ppm`)
original = read_ppm(ppm)
outputs = joinpath(root, "outputs")
write_ppm(joinpath(outputs, "randomized_original.ppm"), original)
exact = compress_image(original, 20)
optimal = image_metrics(original, exact, 20)
write_ppm(joinpath(outputs, "randomized_optimal.ppm"), exact)
for iterations in 0:2
    compressed = compress_image(original, 20; algorithm=:randomized, power_iterations=iterations, rng=MersenneTwister(17))
    metrics = image_metrics(original, compressed, 20)
    path = joinpath(outputs, "randomized_q$(iterations).ppm")
    write_ppm(path, compressed)
    @printf("RGB 512x512 rank=20 q=%d PSNR=%.2f dB optimal=%.2f dB output=%s\n", iterations, metrics.psnr, optimal.psnr, path)
end
