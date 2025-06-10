using Test
using BenchmarkTools
using SpidersMessageCodecs
using LinearAlgebra

@testset "Benchmark Tests" begin
    
    @testset "Performance Regression Tests" begin
        # These tests ensure performance doesn't regress significantly
        
        @testset "Small Tensor Encoding" begin
            data = Float32[1.0 2.0; 3.0 4.0]
            buf = zeros(UInt8, 1000)
            
            result = @benchmark begin
                position_ptr = Ref{Int64}(0)
                enc = TensorMessageEncoder($buf; position_ptr)
                encode(enc, $data)
            end samples=100 evals=10
            
            # Performance assertions
            @test minimum(result).time < 10_000  # Less than 10μs
            @test minimum(result).allocs <= 5    # Minimal allocations
            
            println("Small tensor encoding: $(minimum(result))")
        end
        
        @testset "Medium Tensor Encoding/Decoding" begin
            data = rand(Float32, 100, 100)
            buf = zeros(UInt8, 50_000)
            
            # Test encoding
            encode_result = @benchmark begin
                position_ptr = Ref{Int64}(0)
                enc = TensorMessageEncoder($buf; position_ptr)
                encode(enc, $data)
            end samples=50 evals=5
            
            @test minimum(encode_result).time < 100_000  # Less than 100μs
            println("Medium tensor encoding: $(minimum(encode_result))")
            
            # Test type-stable decoding
            decode_result = @benchmark begin
                position_ptr = Ref{Int64}(0)
                enc = TensorMessageEncoder($buf; position_ptr)
                encode(enc, $data)
                position_ptr[] = 0
                dec = TensorMessageDecoder($buf; position_ptr)
                decode(dec, Matrix{Float32})
            end samples=50 evals=5
            
            @test minimum(decode_result).time < 200_000  # Less than 200μs
            println("Medium tensor decoding: $(minimum(decode_result))")
        end
        
        @testset "Zero Allocation Encoding" begin
            data = Int32[1 2; 3 4]
            buf = zeros(UInt8, 1000)
            position_ptr = Ref{Int64}(0)
            enc = TensorMessageEncoder(buf; position_ptr)
            
            # Warm up
            encode(enc, data)
            
            # Test zero allocation (after warmup)
            position_ptr[] = 0
            result = @benchmark encode($enc, $data) setup=(position_ptr[] = 0)
            
            # Should be very few allocations after warmup
            @test minimum(result).allocs <= 10
            @test minimum(result).time < 5_000  # Less than 5μs
            
            println("Zero allocation encoding: $(minimum(result))")
        end
        
        @testset "EventMessage Performance" begin
            buf = zeros(UInt8, 1000)
            
            # String encoding
            string_result = @benchmark begin
                position_ptr = Ref{Int64}(0)
                enc = EventMessageEncoder($buf; position_ptr)
                encode(enc, "Hello, World!")
            end samples=100 evals=10
            
            @test minimum(string_result).time < 5_000  # Less than 5μs
            println("String encoding: $(minimum(string_result))")
            
            # Number encoding
            number_result = @benchmark begin
                position_ptr = Ref{Int64}(0)
                enc = EventMessageEncoder($buf; position_ptr)
                encode(enc, 42.0)
            end samples=100 evals=10
            
            @test minimum(number_result).time < 2_000  # Less than 2μs
            println("Number encoding: $(minimum(number_result))")
        end
        
        @testset "Type Stability" begin
            data = rand(Float32, 10, 10)
            buf = zeros(UInt8, 5000)
            
            # Measure type-stable decode vs dynamic decode
            typestable_result = @benchmark begin
                position_ptr = Ref{Int64}(0)
                enc = TensorMessageEncoder($buf; position_ptr)
                encode(enc, $data)
                position_ptr[] = 0
                dec = TensorMessageDecoder($buf; position_ptr)
                decode(dec, Matrix{Float32})
            end samples=50 evals=5
            
            dynamic_result = @benchmark begin
                position_ptr = Ref{Int64}(0)
                enc = TensorMessageEncoder($buf; position_ptr)
                encode(enc, $data)
                position_ptr[] = 0
                decode($buf; position_ptr)
            end samples=50 evals=5
            
            # Type-stable should be faster or similar
            typestable_time = minimum(typestable_result).time
            dynamic_time = minimum(dynamic_result).time
            
            @test typestable_time <= dynamic_time * 1.5  # At most 50% slower
            
            println("Type-stable decode: $(minimum(typestable_result))")
            println("Dynamic decode: $(minimum(dynamic_result))")
        end
    end
    
    @testset "Memory Efficiency Tests" begin
        @testset "Buffer Reuse" begin
            buf = zeros(UInt8, 10000)
            data1 = Float32[1.0 2.0; 3.0 4.0]
            data2 = Int16[10 20; 30 40]
            
            # Test that reusing the same buffer is efficient
            result = @benchmark begin
                # Encode first message
                position_ptr = Ref{Int64}(0)
                enc1 = TensorMessageEncoder($buf; position_ptr)
                encode(enc1, $data1)
                
                # Encode second message (reuse buffer)
                position_ptr[] = 0
                enc2 = TensorMessageEncoder($buf; position_ptr)
                encode(enc2, $data2)
            end samples=100 evals=10
            
            @test minimum(result).allocs <= 20  # Should be very few allocations
            println("Buffer reuse: $(minimum(result))")
        end
        
        @testset "Large Data Scaling" begin
            # Test that performance scales reasonably with data size
            sizes = [(10, 10), (50, 50), (100, 100)]
            times = Float64[]
            
            for (rows, cols) in sizes
                data = rand(Float32, rows, cols)
                buf = zeros(UInt8, rows * cols * 8)  # Generous buffer
                
                result = @benchmark begin
                    position_ptr = Ref{Int64}(0)
                    enc = TensorMessageEncoder($buf; position_ptr)
                    encode(enc, $data)
                end samples=20 evals=5
                
                push!(times, minimum(result).time)
            end
            
            # Performance should scale sub-quadratically with data size
            # (since data copying should dominate, which is linear in elements)
            ratio_1_2 = times[2] / times[1]  # 50x50 vs 10x10 (25x more data)
            ratio_2_3 = times[3] / times[2]  # 100x100 vs 50x50 (4x more data)
            
            @test ratio_1_2 < 50    # Should be much less than 50x slower
            @test ratio_2_3 < 10    # Should be much less than 10x slower
            
            println("Scaling times: $(times)")
        end
    end
end
