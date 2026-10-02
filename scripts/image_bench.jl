using Printf
using SVDScratch

function pipeline(path::AbstractString, output_directory::AbstractString)
    original = read_ppm(path)
    metrics = Vector{NamedTuple}(undef, 3)
    elapsed = @elapsed begin
        ranks = (5, 20, 50)
        compressed = compress_image(original, ranks)
        for (index, rank) in enumerate(ranks)
            metrics[index] = image_metrics(original, compressed[index], rank)
            write_ppm(joinpath(output_directory, "$(basename(path))_rank_$(rank).ppm"), compressed[index])
        end
    end
    elapsed, size(original[1]), metrics
end

length(ARGS) > 0 || error("provide one or more PPM images")
output_directory = joinpath(tempdir(), "svd_image_bench_julia")
mkpath(output_directory)
pipeline(first(ARGS), output_directory)
for path in ARGS
    elapsed, (rows, columns), metrics = pipeline(path, output_directory)
    @printf("image=%s size=%zux%zu julia_warm=%.6f s rank20_psnr=%.2f dB\n", basename(path), rows, columns, elapsed, metrics[2].psnr)
end
