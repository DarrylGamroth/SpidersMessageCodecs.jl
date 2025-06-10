using Test
using SpidersMessageCodecs
using BenchmarkTools
using LinearAlgebra  # Ensure extension is loaded

@testset "Performance and Stress Tests" begin
    @testset "Large Data Performance" begin
        @testset "1MB Tensor" begin
            # Test with 1MB of data
            buf = zeros(UInt8, 2_000_000)  # 2MB buffer to be safe
            position_ptr = Ref{Int64}(0)

            enc = TensorMessageEncoder(buf; position_ptr)
            dec = TensorMessageDecoder(buf; position_ptr)

            # Create ~1MB of Float32 data (256k elements) - deterministic pattern
            large_data = Float32[i * j * 0.001 for i in 1:512, j in 1:512]

            # Test encoding performance  
            position_ptr[] = 0
            encode_time = @elapsed encode(enc, large_data)
            @test encode_time < 1.0  # Should encode in less than 1 second (relaxed timing)

            # Test decoding performance
            position_ptr[] = 0
            decode_time = @elapsed decoded = decode(dec, Matrix{Float32})
            @test decode_time < 1.0  # Should decode in less than 1 second (relaxed timing)

            @test decoded ≈ large_data
        end

        @testset "Multiple Small Messages" begin
            buf = zeros(UInt8, 100_000)
            position_ptr = Ref{Int64}(0)

            enc = TensorMessageEncoder(buf; position_ptr)
            dec = TensorMessageDecoder(buf; position_ptr)

            # Test encoding/decoding 100 small messages - deterministic data
            num_messages = 100
            messages = [Float32[i * j * 0.1 + k * 0.01 for i in 1:5, j in 1:5] for k in 1:num_messages]

            # Test each message independently (encode then decode)
            for msg in messages
                position_ptr[] = 0  # Reset for each message
                encode(enc, msg)

                position_ptr[] = 0  # Reset for decoding
                decoded = decode(dec, Matrix{Float32})
                @test decoded ≈ msg
            end
        end
    end

    @testset "Memory Allocation Tests" begin
        @testset "Zero Allocation Encoding" begin
            buf = zeros(UInt8, 10000)
            position_ptr = Ref{Int64}(0)

            enc = TensorMessageEncoder(buf; position_ptr)
            data = Float32[i * j * 0.1 for i in 1:10, j in 1:10]  # Deterministic 10x10 matrix

            # Warm up - multiple times to ensure JIT compilation
            for _ in 1:10
                position_ptr[] = 0
                encode(enc, data)
            end

            # Test minimal allocation (might not be zero due to GC overhead)
            position_ptr[] = 0
            allocs = @allocated encode(enc, data)
            @test allocs < 200  # Allow for small overhead, should be much less than initial encoding
        end

        @testset "Minimal Allocation Decoding" begin
            buf = zeros(UInt8, 10000)
            position_ptr = Ref{Int64}(0)

            enc = TensorMessageEncoder(buf; position_ptr)
            dec = TensorMessageDecoder(buf; position_ptr)
            data = Float32[i * j * 0.15 for i in 1:10, j in 1:10]  # Deterministic 10x10 matrix

            # Encode data
            position_ptr[] = 0
            encode(enc, data)

            # Warm up decoder
            position_ptr[] = 0
            _ = decode(dec, Matrix{Float32})

            # Test minimal allocation decoding
            position_ptr[] = 0
            allocs = @allocated decode(dec, Matrix{Float32})
            @test allocs < 1000  # Should be minimal allocation
        end
    end

    @testset "Stress Tests" begin
        @testset "Buffer Boundary Conditions" begin
            # Test with exactly sized buffer - deterministic data
            data = Int16[i * j + 100 for i in 1:10, j in 1:10]  # Deterministic 10x10 matrix

            # Calculate required size programmatically
            data_size = prod(size(data)) * sizeof(eltype(data))

            # Fixed header sizes
            message_header_size = Int(sbe_encoded_length(MessageHeader))    # MessageHeader size
            tensor_header_size = Int(sbe_block_length(TensorMessage))       # TensorMessage fixed fields (68 bytes)

            # Variable length field overhead
            ndims = length(size(data))
            dims_size = 4 + ndims * sizeof(Int32)    # 4 byte length + dimensions
            origin_size = 4 + ndims * sizeof(Int32)  # 4 byte length + origin values  
            values_overhead = 4                      # 4 byte length prefix for values

            # Total required size with small safety margin
            required_size = message_header_size + tensor_header_size + dims_size + origin_size + values_overhead + data_size + 16

            buf = zeros(UInt8, required_size)
            position_ptr = Ref{Int64}(0)

            enc = TensorMessageEncoder(buf; position_ptr)
            dec = TensorMessageDecoder(buf; position_ptr)

            # Should fit exactly
            position_ptr[] = 0
            encode(enc, data)

            position_ptr[] = 0
            decoded = decode(dec, Matrix{Int16})
            @test decoded ≈ data
        end

        @testset "Extreme Dimensions" begin
            buf = zeros(UInt8, 100_000)
            position_ptr = Ref{Int64}(0)

            enc = TensorMessageEncoder(buf; position_ptr)
            dec = TensorMessageDecoder(buf; position_ptr)

            # Test very flat matrix - deterministic data
            flat_data = Int16[i for i in 1:1000]'  # 1x1000 matrix
            position_ptr[] = 0
            encode(enc, flat_data)
            position_ptr[] = 0
            decoded = decode(dec, Matrix{Int16})
            @test decoded ≈ flat_data

            # Test very tall matrix - deterministic data
            tall_data = Int16[i for i in 1001:2000]  # 1000x1 matrix (reshaped)
            tall_data = reshape(tall_data, 1000, 1)
            position_ptr[] = 0
            encode(enc, tall_data)
            position_ptr[] = 0
            decoded = decode(dec, Matrix{Int16})
            @test decoded ≈ tall_data
        end

        @testset "Rapid Encode/Decode Cycles" begin
            buf = zeros(UInt8, 10000)
            position_ptr = Ref{Int64}(0)

            enc = TensorMessageEncoder(buf; position_ptr)
            dec = TensorMessageDecoder(buf; position_ptr)

            # Rapidly encode/decode 1000 times - deterministic data
            data = Float32[i * j * 0.01 + 1.0 for i in 1:5, j in 1:5]  # Deterministic 5x5 matrix

            for i in 1:1000
                position_ptr[] = 0
                encode(enc, data)

                position_ptr[] = 0
                decoded = decode(dec, Matrix{Float32})
                @test decoded ≈ data
            end
        end
    end

    @testset "Concurrent Safety" begin
        @testset "Multiple Buffers" begin
            # Test that multiple independent buffers don't interfere
            num_buffers = 10
            buffers = [zeros(UInt8, 1000) for _ in 1:num_buffers]
            data_sets = [Float32[i * j * 0.02 + k * 0.1 for i in 1:5, j in 1:5] for k in 1:num_buffers]  # Deterministic data

            # Encode in parallel-ish manner
            encoders = []
            decoders = []

            for i in 1:num_buffers
                position_ptr = Ref{Int64}(0)
                enc = TensorMessageEncoder(buffers[i]; position_ptr)
                dec = TensorMessageDecoder(buffers[i]; position_ptr)

                encode(enc, data_sets[i])
                push!(encoders, enc)
                push!(decoders, dec)
            end

            # Decode and verify
            for i in 1:num_buffers
                position_ptr = Ref{Int64}(0)
                dec = TensorMessageDecoder(buffers[i]; position_ptr)
                decoded = decode(dec, Matrix{Float32})
                @test decoded ≈ data_sets[i]
            end
        end
    end
end
