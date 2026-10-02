using Printf
using SVDScratch

original = read_ppm("assets/real_landscape.ppm")
ranks = (5, 20, 50)
for (rank, compressed) in zip(ranks, compress_image(original, ranks))
    metrics = image_metrics(original, compressed, rank)
    path = "outputs/landscape_rank_$(rank).ppm"
    write_ppm(path, compressed)
    @printf("rank=%d MSE=%.6e PSNR=%.2f dB scalar_ratio=%.2fx output=%s\n", rank, metrics.mse, metrics.psnr, metrics.scalar_ratio, path)
end
