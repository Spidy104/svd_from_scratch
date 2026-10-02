export read_ppm, write_ppm, compress_image, image_metrics

function ppm_header(data::Vector{UInt8})
    index = 1
    tokens = String[]
    while length(tokens) < 4
        while index <= length(data) && isspace(Char(data[index]))
            index += 1
        end
        index <= length(data) || throw(ArgumentError("incomplete PPM header"))
        if data[index] == UInt8('#')
            while index <= length(data) && data[index] != UInt8('\n')
                index += 1
            end
            continue
        end
        start = index
        while index <= length(data) && !isspace(Char(data[index]))
            index += 1
        end
        push!(tokens, String(data[start:(index - 1)]))
    end
    index <= length(data) && isspace(Char(data[index])) || throw(ArgumentError("invalid PPM header"))
    delimiter = data[index]
    index += 1
    if delimiter == UInt8('\r') && index <= length(data) && data[index] == UInt8('\n')
        index += 1
    end
    tokens, index
end

function read_ppm(path::AbstractString)
    data = read(path)
    tokens, index = ppm_header(data)
    tokens[1] == "P6" || throw(ArgumentError("only binary PPM images are supported"))
    width = parse(Int, tokens[2])
    height = parse(Int, tokens[3])
    maximum = parse(Int, tokens[4])
    maximum == 255 || throw(ArgumentError("only 8-bit PPM images are supported"))
    width > 0 && height > 0 && height <= typemax(Int) ÷ 3 && width <= typemax(Int) ÷ (3 * height) || throw(ArgumentError("invalid PPM dimensions"))
    length(data) - index + 1 == 3 * width * height || throw(ArgumentError("invalid PPM payload"))
    channels = [Matrix{Float64}(undef, height, width) for _ in 1:3]
    offset = index
    @inbounds for row in 1:height
        for column in 1:width
            for channel in 1:3
                channels[channel][row, column] = data[offset] / 255.0
                offset += 1
            end
        end
    end
    channels
end

function image_size(channels::Vector{Matrix{Float64}})
    length(channels) == 3 || throw(ArgumentError("three RGB channels are required"))
    height, width = size(channels[1])
    height > 0 && width > 0 || throw(ArgumentError("image must be nonempty"))
    all(size(channel) == (height, width) for channel in channels) || throw(DimensionMismatch("channel dimensions differ"))
    height, width
end

function write_ppm(path::AbstractString, channels::Vector{Matrix{Float64}})
    height, width = image_size(channels)
    buffer = Vector{UInt8}(undef, 3 * width)
    open(path, "w") do io
        write(io, "P6\n$(width) $(height)\n255\n")
        @inbounds for row in 1:height
            for column in 1:width
                for channel in 1:3
                    buffer[3 * (column - 1) + channel] = UInt8(round(Int, 255 * clamp(channels[channel][row, column], 0.0, 1.0)))
                end
            end
            write(io, buffer)
        end
    end
    path
end

function rank_reconstruction(factor, rank::Integer)
    scaled = Matrix{eltype(factor.U)}(undef, size(factor.U, 1), rank)
    @inbounds for j in 1:rank, i in axes(scaled, 1)
        scaled[i, j] = factor.U[i, j] * factor.S[j]
    end
    scaled * adjoint(view(factor.V, :, 1:rank))
end

function compress_image(channels::Vector{Matrix{Float64}}, rank::Integer; algorithm::Symbol=:auto, kwargs...)
    rank > 0 || throw(ArgumentError("rank must be positive"))
    height, width = image_size(channels)
    rank <= min(height, width) || throw(ArgumentError("rank exceeds image dimensions"))
    compressed = Vector{Matrix{Float64}}(undef, 3)
    for channel in 1:3
        F = algorithm === :randomized ? randomized_svd(channels[channel], rank; kwargs...) : svd_scratch(channels[channel]; algorithm, kwargs...)
        F.converged || error("image SVD did not converge")
        compressed[channel] = rank_reconstruction(F, rank)
    end
    compressed
end

function compress_image(channels::Vector{Matrix{Float64}}, ranks::Tuple{Vararg{Int}}; algorithm::Symbol=:auto, kwargs...)
    height, width = image_size(channels)
    !isempty(ranks) && all(rank -> 1 <= rank <= min(height, width), ranks) || throw(ArgumentError("invalid image ranks"))
    results = [Vector{Matrix{Float64}}(undef, 3) for _ in ranks]
    for channel in 1:3
        F = algorithm === :randomized ? randomized_svd(channels[channel], maximum(ranks); kwargs...) : svd_scratch(channels[channel]; algorithm, kwargs...)
        F.converged || error("image SVD did not converge")
        for (index, rank) in enumerate(ranks)
            results[index][channel] = rank_reconstruction(F, rank)
        end
    end
    results
end

function image_metrics(original::Vector{Matrix{Float64}}, compressed::Vector{Matrix{Float64}}, rank::Integer)
    height, width = image_size(original)
    image_size(compressed) == (height, width) || throw(DimensionMismatch("image dimensions differ"))
    1 <= rank <= min(height, width) || throw(ArgumentError("invalid image rank"))
    total = 0.0
    @inbounds for channel in 1:3
        for i in eachindex(original[channel])
            difference = original[channel][i] - compressed[channel][i]
            total += difference * difference
        end
    end
    mse = total / (3 * height * width)
    (mse=mse, psnr=mse == 0.0 ? Inf : 10.0 * log10(inv(mse)), scalar_ratio=(height * width) / (rank * (height + width + 1)))
end
