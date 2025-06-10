using Test
using SpidersMessageCodecs
using LinearAlgebra

@testset "Core TensorMessage and EventMessage Tests" begin
    
    @testset "TensorMessage Type-Stable Encoding/Decoding" begin
        @testset "Basic Data Types" begin
            buf = zeros(UInt8, 10000)  # Larger buffer for comprehensive tests
            position_ptr = Ref{Int64}(0)
            
            enc = TensorMessageEncoder(buf; position_ptr)
            dec = TensorMessageDecoder(buf; position_ptr)
            
            # Test with different matrix types and sizes - deterministic data
            test_data = [
                Int8[1 2 3; 4 5 6],                                    # Int8 2x3
                UInt8[10 20 30 40 50; 60 70 80 90 100; 110 120 130 140 150; 160 170 180 190 200], # UInt8 4x5
                Int16[100 200 300 400 500 600 700 800; 900 1000 1100 1200 1300 1400 1500 1600; 1700 1800 1900 2000 2100 2200 2300 2400; 2500 2600 2700 2800 2900 3000 3100 3200; 3300 3400 3500 3600 3700 3800 3900 4000], # Int16 5x8
                UInt16[1000 2000 3000 4000 5000 6000; 7000 8000 9000 10000 11000 12000; 13000 14000 15000 16000 17000 18000], # UInt16 3x6
                Int32[100000 200000 300000 400000 500000 600000; 700000 800000 900000 1000000 1100000 1200000], # Int32 2x6
                UInt32[1000000 2000000 3000000 4000000; 5000000 6000000 7000000 8000000; 9000000 10000000 11000000 12000000], # UInt32 3x4
                Int64[1000000000 2000000000 3000000000; 4000000000 5000000000 6000000000], # Int64 2x3
                UInt64[10000000000 20000000000 30000000000; 40000000000 50000000000 60000000000], # UInt64 2x3
                Float32[1.5 2.5 3.5 4.5; 5.5 6.5 7.5 8.5; 9.5 10.5 11.5 12.5], # Float32 3x4
                Float64[1.25 2.25 3.25 4.25 5.25; 6.25 7.25 8.25 9.25 10.25], # Float64 2x5
            ]
            
            for data in test_data
                position_ptr[] = 0  # Reset position
                
                # Encode
                encode(enc, data)
                
                # Decode with type specification (type-stable)
                position_ptr[] = 0  # Reset for decoding
                decoded = decode(dec, typeof(data))
                
                @test decoded ≈ data
                @test size(decoded) == size(data)
                @test eltype(decoded) == eltype(data)
            end
        end
        
        @testset "Different Dimensionalities" begin
            buf = zeros(UInt8, 10000)
            position_ptr = Ref{Int64}(0)
            
            enc = TensorMessageEncoder(buf; position_ptr)
            dec = TensorMessageDecoder(buf; position_ptr)
            
            # Test vectors, matrices, and higher-dimensional arrays - deterministic data
            test_arrays = [
                Float32[1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10],  # 1D vector
                Int16[10 20 30 40 50 60 70 80; 90 100 110 120 130 140 150 160; 170 180 190 200 210 220 230 240; 250 260 270 280 290 300 310 320; 330 340 350 360 370 380 390 400], # 2D matrix 5x8
                reshape(Float64[i*0.5 for i in 1:24], 2, 3, 4),  # 3D array 2x3x4
                reshape(Int32[i*100 for i in 1:16], 2, 2, 2, 2), # 4D array 2x2x2x2
            ]
            
            for data in test_arrays
                position_ptr[] = 0
                encode(enc, data)
                
                position_ptr[] = 0
                decoded = decode(dec, typeof(data))
                
                @test decoded ≈ data
                @test size(decoded) == size(data)
                @test ndims(decoded) == ndims(data)
            end
        end
        
        @testset "Edge Cases and Boundary Conditions" begin
            buf = zeros(UInt8, 10000)
            position_ptr = Ref{Int64}(0)
            
            enc = TensorMessageEncoder(buf; position_ptr)
            dec = TensorMessageDecoder(buf; position_ptr)
            
            # Test with minimal arrays - deterministic data
            minimal_data = [
                Float32[3.14],                       # Single element matrix
                Int16[42],                           # Single element vector
                zeros(Int32, 2, 3),                  # All zeros
                ones(Float64, 3, 2),                 # All ones
            ]
            
            for data in minimal_data
                position_ptr[] = 0
                encode(enc, data)
                
                position_ptr[] = 0
                decoded = decode(dec, typeof(data))
                
                @test decoded ≈ data
                @test size(decoded) == size(data)
            end
        end
        
        @testset "Large Arrays" begin
            buf = zeros(UInt8, 1_000_000)  # 1MB buffer
            position_ptr = Ref{Int64}(0)
            
            enc = TensorMessageEncoder(buf; position_ptr)
            dec = TensorMessageDecoder(buf; position_ptr)
            
            # Test with larger arrays - deterministic pattern
            large_data = Float32[i*j*0.01 for i in 1:100, j in 1:100]
            
            position_ptr[] = 0
            encode(enc, large_data)
            
            position_ptr[] = 0
            decoded = decode(dec, Matrix{Float32})
            
            @test decoded ≈ large_data
            @test size(decoded) == size(large_data)
        end
    end
    
    @testset "EventMessage Encoding/Decoding" begin
        @testset "String Values" begin
            buf = zeros(UInt8, 1000)
            position_ptr = Ref{Int64}(0)
            
            enc = EventMessageEncoder(buf; position_ptr)
            dec = EventMessageDecoder(buf; position_ptr)
            
            test_strings = [
                "Hello, World!",
                "",  # Empty string
                "Unicode: αβγδε 🚀 ñáéíóú",
                "A" ^ 100,  # Long string
            ]
            
            for test_value in test_strings
                position_ptr[] = 0
                encode(enc, test_value)
                
                position_ptr[] = 0
                decoded = decode(dec, String)
                @test decoded == test_value
            end
        end
        
        @testset "Numeric Values" begin
            buf = zeros(UInt8, 1000)
            position_ptr = Ref{Int64}(0)
            
            enc = EventMessageEncoder(buf; position_ptr)
            dec = EventMessageDecoder(buf; position_ptr)
            
            test_numbers = [
                Int8(42),
                UInt8(255),
                Int16(-1000),
                UInt16(65535),
                Int32(-1_000_000),
                UInt32(4_000_000_000),
                Int64(-9_223_372_036_854_775_808),
                UInt64(18_446_744_073_709_551_615),
                Float32(3.14159),
                Float64(2.718281828459045),
            ]
            
            for test_value in test_numbers
                position_ptr[] = 0
                encode(enc, test_value)
                
                position_ptr[] = 0
                decoded = decode(dec, typeof(test_value))
                @test decoded ≈ test_value
            end
        end
        
        @testset "Boolean Values" begin
            buf = zeros(UInt8, 1000)
            position_ptr = Ref{Int64}(0)
            
            enc = EventMessageEncoder(buf; position_ptr)
            dec = EventMessageDecoder(buf; position_ptr)
            
            for test_value in [true, false]
                position_ptr[] = 0
                encode(enc, test_value)
                
                position_ptr[] = 0
                decoded = decode(dec, Bool)
                @test decoded == test_value
            end
        end
    end
    
    @testset "Dynamic Decoding" begin
        @testset "Type-Unstable Decoding" begin
            buf = zeros(UInt8, 1000)
            position_ptr = Ref{Int64}(0)
            
            enc = TensorMessageEncoder(buf; position_ptr)
            data = Int16[10 20 30 40; 50 60 70 80; 90 100 110 120]  # Deterministic 3x4 matrix
            
            encode(enc, data)
            
            # Test type-unstable dynamic decoding
            position_ptr[] = 0
            decoded_tensor = decode(buf; position_ptr)
            actual_decoded = decode(decoded_tensor, Matrix{Int16})
            @test actual_decoded ≈ data
        end
        
        @testset "Format Detection" begin
            buf = zeros(UInt8, 1000)
            position_ptr = Ref{Int64}(0)
            
            enc = TensorMessageEncoder(buf; position_ptr)
            dec = TensorMessageDecoder(buf; position_ptr)
            
            # Test different data types and verify format detection - deterministic data
            test_cases = [
                (Int8[1 2 3; 4 5 6], SpidersMessageCodecs.Format.INT8),
                (UInt8[10 20 30; 40 50 60], SpidersMessageCodecs.Format.UINT8),
                (Int16[100 200 300; 400 500 600], SpidersMessageCodecs.Format.INT16),
                (UInt16[1000 2000 3000; 4000 5000 6000], SpidersMessageCodecs.Format.UINT16),
                (Int32[10000 20000 30000; 40000 50000 60000], SpidersMessageCodecs.Format.INT32),
                (UInt32[100000 200000 300000; 400000 500000 600000], SpidersMessageCodecs.Format.UINT32),
                (Int64[1000000 2000000 3000000; 4000000 5000000 6000000], SpidersMessageCodecs.Format.INT64),
                (UInt64[10000000 20000000 30000000; 40000000 50000000 60000000], SpidersMessageCodecs.Format.UINT64),
                (Float32[1.5 2.5 3.5; 4.5 5.5 6.5], SpidersMessageCodecs.Format.FLOAT32),
                (Float64[1.25 2.25 3.25; 4.25 5.25 6.25], SpidersMessageCodecs.Format.FLOAT64),
            ]
            
            for (data, expected_format) in test_cases
                position_ptr[] = 0
                encode(enc, data)
                
                position_ptr[] = 0
                SpidersMessageCodecs.sbe_rewind!(dec)
                detected_format = SpidersMessageCodecs.format(dec)
                @test detected_format == expected_format
            end
        end
    end
    
    @testset "Memory Layout and Major Order" begin
        @testset "Column vs Row Major" begin
            buf = zeros(UInt8, 1000)
            position_ptr = Ref{Int64}(0)
            
            enc = TensorMessageEncoder(buf; position_ptr)
            dec = TensorMessageDecoder(buf; position_ptr)
            
            # Test column-major (Julia default) - deterministic data
            data = Int16[10 20 30 40; 50 60 70 80; 90 100 110 120]  # 3x4 matrix
            position_ptr[] = 0
            encode(enc, data)
            position_ptr[] = 0
            decoded = decode(dec, Matrix{Int16})
            @test decoded == data
            
            # Test row-major (transposed)
            position_ptr[] = 0
            encode(enc, data')
            position_ptr[] = 0
            decoded_t = decode(dec, typeof(data'))
            @test decoded_t == data'
        end
        
        @testset "Major Order Detection" begin
            buf = zeros(UInt8, 1000)
            position_ptr = Ref{Int64}(0)
            
            enc = TensorMessageEncoder(buf; position_ptr)
            dec = TensorMessageDecoder(buf; position_ptr)
            
            # Test column-major detection - deterministic data
            data_col = Float32[1.5 2.5 3.5 4.5; 5.5 6.5 7.5 8.5; 9.5 10.5 11.5 12.5]  # 3x4 matrix
            position_ptr[] = 0
            encode(enc, data_col)
            position_ptr[] = 0
            SpidersMessageCodecs.sbe_rewind!(dec)
            major_order = SpidersMessageCodecs.majorOrder(dec)
            @test major_order == SpidersMessageCodecs.MajorOrder.COLUMN
            
            # Test row-major detection (transposed)
            data_row = data_col'
            position_ptr[] = 0
            encode(enc, data_row)
            position_ptr[] = 0
            SpidersMessageCodecs.sbe_rewind!(dec)
            major_order = SpidersMessageCodecs.majorOrder(dec)
            @test major_order == SpidersMessageCodecs.MajorOrder.ROW
        end
    end
    
    @testset "Buffer Management and Safety" begin
        @testset "Position Pointer Handling" begin
            buf = zeros(UInt8, 1000)
            position_ptr = Ref{Int64}(0)
            
            enc = TensorMessageEncoder(buf; position_ptr)
            data1 = Int16[10 20 30; 40 50 60]                        # 2x3 matrix
            
            # Test encoding and position advancement
            encode(enc, data1)
            pos_after_encode = position_ptr[]
            @test pos_after_encode > 0
            
            # Test decoding and position management
            position_ptr[] = 0  # Reset for decoding
            dec = TensorMessageDecoder(buf; position_ptr)
            decoded1 = decode(dec, Matrix{Int16})
            @test decoded1 ≈ data1
            
            pos_after_decode = position_ptr[]
            @test pos_after_decode == pos_after_encode  # Should be at same position
        end
        
        @testset "Buffer Reuse" begin
            buf = zeros(UInt8, 1000)
            position_ptr = Ref{Int64}(0)
            
            enc = TensorMessageEncoder(buf; position_ptr)
            dec = TensorMessageDecoder(buf; position_ptr)
            
            # Encode multiple messages in the same buffer - deterministic data
            test_data = [
                Int16[10 20; 30 40],                                 # 2x2 matrix
                Float32[1.5 2.5 3.5; 4.5 5.5 6.5; 7.5 8.5 9.5],    # 3x3 matrix
                Int32[100 200 300 400 500],                         # 1x5 matrix
            ]
            
            for data in test_data
                position_ptr[] = 0  # Reset for each message
                encode(enc, data)
                
                position_ptr[] = 0  # Reset for decoding
                decoded = decode(dec, typeof(data))
                @test decoded ≈ data
            end
        end
    end
    
    @testset "Origin Support" begin
        @testset "Tensor Origins" begin
            buf = zeros(UInt8, 1000)
            position_ptr = Ref{Int64}(0)
            
            enc = TensorMessageEncoder(buf; position_ptr)
            dec = TensorMessageDecoder(buf; position_ptr)
            
            data = Float32[1.5 2.5 3.5 4.5; 5.5 6.5 7.5 8.5; 9.5 10.5 11.5 12.5]  # 3x4 matrix
            origin = (Int32(10), Int32(20))  # Ensure correct type
            
            position_ptr[] = 0
            encode(enc, data; origin=origin)
            
            position_ptr[] = 0
            decoded = decode(dec, Matrix{Float32})
            @test decoded ≈ data
            
            # Test origin separately - the API might need investigation
            # For now, just test that encoding/decoding works with origin parameter
            @test size(decoded) == size(data)
        end
    end
    
    @testset "Error Handling and Validation" begin
        @testset "Type Validation" begin
            buf = zeros(UInt8, 1000)
            position_ptr = Ref{Int64}(0)
            
            enc = TensorMessageEncoder(buf; position_ptr)
            dec = TensorMessageDecoder(buf; position_ptr)
            
            # Enable validation
            old_val = get(ENV, "SPIDERS_VALIDATE_TYPES", "false")
            ENV["SPIDERS_VALIDATE_TYPES"] = "true"
            
            try
                data = Int16[10 20 30 40; 50 60 70 80; 90 100 110 120]  # 3x4 matrix
                position_ptr[] = 0
                encode(enc, data)
                
                position_ptr[] = 0
                # This should work fine
                decoded = decode(dec, Matrix{Int16})
                @test decoded ≈ data
                
                position_ptr[] = 0
                # This should throw an error due to type mismatch
                @test_throws DimensionMismatch decode(dec, Matrix{Float32})
            finally
                if old_val == "false"
                    delete!(ENV, "SPIDERS_VALIDATE_TYPES")
                else
                    ENV["SPIDERS_VALIDATE_TYPES"] = old_val
                end
            end
        end
        
        @testset "Dimension Validation" begin
            buf = zeros(UInt8, 1000)
            position_ptr = Ref{Int64}(0)
            
            enc = TensorMessageEncoder(buf; position_ptr)
            dec = TensorMessageDecoder(buf; position_ptr)
            
            data = rand(Float32, 3, 4)
            position_ptr[] = 0
            encode(enc, data)
            
            position_ptr[] = 0
            # This should work
            decoded = decode(dec, Matrix{Float32})
            @test size(decoded) == size(data)
            
            position_ptr[] = 0
            # This should fail - wrong dimensions
            @test_throws Exception decode(dec, Array{Float32, 3})
        end
    end
    
    @testset "Special Values" begin
        @testset "IEEE Float Special Values" begin
            buf = zeros(UInt8, 1000)
            position_ptr = Ref{Int64}(0)
            
            enc = TensorMessageEncoder(buf; position_ptr)
            dec = TensorMessageDecoder(buf; position_ptr)
            
            # Test special floating point values
            special_values = Float32[Inf, -Inf, NaN, 0.0, -0.0, 
                                   floatmin(Float32), floatmax(Float32)]
            data = reshape(special_values, 1, length(special_values))
            
            position_ptr[] = 0
            encode(enc, data)
            
            position_ptr[] = 0
            decoded = decode(dec, Matrix{Float32})
            
            # Test each value individually (NaN != NaN)
            for i in 1:length(special_values)
                if isnan(special_values[i])
                    @test isnan(decoded[1, i])
                else
                    @test decoded[1, i] == special_values[i]
                end
            end
        end
        
        @testset "Integer Boundary Values" begin
            buf = zeros(UInt8, 1000)
            position_ptr = Ref{Int64}(0)
            
            enc = TensorMessageEncoder(buf; position_ptr)
            dec = TensorMessageDecoder(buf; position_ptr)
            
            # Test integer boundary values
            int32_boundaries = Int32[typemin(Int32), -1, 0, 1, typemax(Int32)]
            data = reshape(int32_boundaries, 1, length(int32_boundaries))
            
            position_ptr[] = 0
            encode(enc, data)
            
            position_ptr[] = 0
            decoded = decode(dec, Matrix{Int32})
            @test decoded == data
        end
    end
end
