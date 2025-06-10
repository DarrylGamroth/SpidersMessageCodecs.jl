#!/usr/bin/env julia

"""
Comprehensive test for zero-allocation decoding solutions
"""

using SpidersMessageCodecs
using FixedSizeArrays
using LinearAlgebra

# Include the zero-allocation module
include("../ext/zero_alloc.jl")

function test_zero_alloc_solutions()
    println("🚀 Testing Zero-Allocation Decoding Solutions")
    println("=" ^ 60)
    
    # Setup test data
    buf = zeros(UInt8, 2000)
    position_ptr = Ref{Int64}(0)
    test_data = Float32[1.0 2.0 3.0; 4.0 5.0 6.0]  # 2x3 matrix
    
    # Encode test data
    enc = TensorMessageEncoder(buf; position_ptr)
    encode(enc, test_data)
    encoded_size = position_ptr[]
    
    println("✓ Test data encoded: $(size(test_data)) matrix, $encoded_size bytes")
    println()
    
    # Test 1: Fixed-size array decoder (FixedSizeArrays.jl)
    println("🧪 Test 1: FixedSizeArray Decoder")
    
    function test_fixed_size()
        position_ptr[] = 0
        dec = TensorMessageDecoder(buf; position_ptr)
        return decode_stack(FixedSizeMatrix{Float32}, dec, 2, 3)
    end
    
    # Warm up
    result1 = test_fixed_size()
    
    # Test allocation
    allocs1 = @allocated test_fixed_size()
    println("   Allocations: $allocs1 bytes")
    println("   Result matches: $(Matrix(result1) ≈ test_data)")
    println("   Result type: $(typeof(result1))")
    println("   Memory backend: $(typeof(parent(result1)))")
    
    if allocs1 == 0
        println("   ✅ ZERO ALLOCATION ACHIEVED!")
    else
        println("   ⚠️  Still allocating $allocs1 bytes")
    end
    println()
    
    # Test 2: User-provided output array
    println("🧪 Test 2: decode_into! with user array")
    output = Matrix{Float32}(undef, 2, 3)
    
    function test_decode_into()
        position_ptr[] = 0
        dec = TensorMessageDecoder(buf; position_ptr)
        return decode_into!(output, dec)
    end
    
    # Warm up
    result2 = test_decode_into()
    
    # Test allocation
    allocs2 = @allocated test_decode_into()
    println("   Allocations: $allocs2 bytes")
    println("   Result matches: $(result2 ≈ test_data)")
    println("   Result is same object: $(result2 === output)")
    
    if allocs2 == 0
        println("   ✅ ZERO ALLOCATION ACHIEVED!")
    else
        println("   ⚠️  Still allocating $allocs2 bytes")
    end
    println()
    
    # Test 3: Raw pointer access (for reference - advanced users only)
    println("🧪 Test 3: Raw Pointer Access (⚠️ Advanced users only)")
    
    function test_pointer()
        position_ptr[] = 0
        dec = TensorMessageDecoder(buf; position_ptr)
        data_ptr, (rows, cols) = decode_unsafe(Matrix{Float32}, dec, 2, 3)
        return data_ptr, rows, cols
    end
    
    # Warm up
    ptr_result, rows, cols = test_pointer()
    
    # Test allocation
    allocs3 = @allocated test_pointer()
    println("   Allocations: $allocs3 bytes")
    println("   Returns: Raw pointer + dimensions")
    println("   ⚠️  DANGER: Caller must manage memory lifetime")
    
    if allocs3 == 0
        println("   ✅ ZERO ALLOCATION ACHIEVED!")
    else
        println("   ⚠️  Still allocating $allocs3 bytes")
    end
    println()
    
    # Test 4: Streaming processing (no arrays)
    println("🧪 Test 4: Streaming Processing (no arrays)")
    
    function test_streaming()
        position_ptr[] = 0
        dec = TensorMessageDecoder(buf; position_ptr)
        return fold_elements(+, 0.0f0, dec, Float32)
    end
    
    # Warm up
    stream_sum = test_streaming()
    
    # Test allocation
    allocs4 = @allocated test_streaming()
    println("   Allocations: $allocs4 bytes")
    println("   Sum result: $stream_sum")
    println("   Expected: $(sum(test_data))")
    println("   Match: $(stream_sum ≈ sum(test_data))")
    
    if allocs4 == 0
        println("   ✅ ZERO ALLOCATION ACHIEVED!")
    else
        println("   ⚠️  Still allocating $allocs4 bytes")
    end
    println()
    
    # Performance comparison
    println("⚡ Performance Comparison")
    println("-" ^ 40)
    
    # Standard decode (for comparison)
    function test_standard()
        position_ptr[] = 0
        dec = TensorMessageDecoder(buf; position_ptr)
        return decode(dec, Matrix{Float32})
    end
    
    result_std = test_standard()
    allocs_std = @allocated test_standard()
    
    println("Standard decode:     $allocs_std bytes")
    println("FixedSizeArray:      $allocs1 bytes ($(allocs_std - allocs1) bytes saved)")
    println("decode_into!:        $allocs2 bytes ($(allocs_std - allocs2) bytes saved)")
    println("Raw pointers:        $allocs3 bytes ($(allocs_std - allocs3) bytes saved)")
    println("Stream processing:   $allocs4 bytes ($(allocs_std - allocs4) bytes saved)")
    println()
    
    # Summary
    zero_alloc_methods = String[]
    if allocs1 == 0; push!(zero_alloc_methods, "FixedSizeArrays"); end
    if allocs2 == 0; push!(zero_alloc_methods, "decode_into!"); end
    if allocs3 == 0; push!(zero_alloc_methods, "Raw pointers"); end
    if allocs4 == 0; push!(zero_alloc_methods, "Stream processing"); end
    
    println("🎯 Summary")
    println("-" ^ 20)
    if !isempty(zero_alloc_methods)
        println("✅ Zero-allocation methods: $(join(zero_alloc_methods, ", "))")
        println("🚀 Ready for garbage collection sensitive applications!")
    else
        println("⚠️  No methods achieved true zero allocation")
        println("   This may be due to Julia version or compiler optimizations")
    end
    
    return zero_alloc_methods
end

# Benchmark function for real-world usage patterns
function benchmark_real_world_usage()
    println("\n" * "=" ^ 60)
    println("📊 Real-World Usage Benchmark")
    println("=" ^ 60)
    
    # Setup for high-frequency trading scenario
    buf = zeros(UInt8, 50000)  # Larger buffer for bigger matrix
    position_ptr = Ref{Int64}(0)
    
    # Typical market data: smaller for benchmark stability
    market_data = rand(Float64, 50, 10)  # 50x10 instead of 100x10
    
    # Encode once
    enc = TensorMessageEncoder(buf; position_ptr)
    encode(enc, market_data)
    
    println("Scenario: High-frequency market data processing")
    println("Data size: $(size(market_data)) Float64 matrix")
    println("Processing 10,000 messages...")
    println()
    
    # Method 1: Standard decode (baseline)
    function process_standard()
        total_allocs = 0
        for i in 1:10000
            position_ptr[] = 0
            dec = TensorMessageDecoder(buf; position_ptr)
            total_allocs += @allocated decode(dec, Matrix{Float64})
        end
        return total_allocs
    end
    
    # Method 2: User-provided array
    user_output = Matrix{Float64}(undef, 50, 10)
    function process_into()
        total_allocs = 0
        for i in 1:10000
            position_ptr[] = 0
            dec = TensorMessageDecoder(buf; position_ptr)
            total_allocs += @allocated decode_into!(user_output, dec)
        end
        return total_allocs
    end
    
    # Run benchmarks
    println("Running benchmarks...")
    
    # Warm up
    process_standard()
    process_into()
    
    # Measure
    time_std = @elapsed allocs_std_total = process_standard()
    time_into = @elapsed allocs_into_total = process_into()
    
    println("Results for 10,000 decode operations:")
    println("-" ^ 40)
    println("Standard decode:")
    println("  Total allocations: $(allocs_std_total) bytes")
    println("  Time: $(round(time_std * 1000, digits=2)) ms")
    println("  Per operation: $(allocs_std_total ÷ 10000) bytes")
    println()
    
    println("decode_into!:")
    println("  Total allocations: $(allocs_into_total) bytes")
    println("  Time: $(round(time_into * 1000, digits=2)) ms")  
    println("  Per operation: $(allocs_into_total ÷ 10000) bytes")
    println("  Memory saved: $(((allocs_std_total - allocs_into_total) / allocs_std_total * 100)) %")
    println()
    
    if allocs_into_total == 0
        println("🎉 ZERO-ALLOCATION DECODING ACHIEVED!")
        println("   Perfect for real-time, GC-sensitive applications")
    else
        gc_pressure_reduction = (allocs_std_total - allocs_into_total) / allocs_std_total * 100
        println("📈 GC Pressure Reduction: $(round(gc_pressure_reduction, digits=1))%")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    zero_alloc_methods = test_zero_alloc_solutions()
    
    if !isempty(zero_alloc_methods)
        benchmark_real_world_usage()
    end
end
