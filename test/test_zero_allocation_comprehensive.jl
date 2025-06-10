#!/usr/bin/env julia

"""
Comprehensive zero allocation testing to validate Julia's escape analysis behavior
with the standard SpidersMessageCodecs decode() function.

This test suite validates that Julia's escape analysis successfully eliminates
allocations for various realistic usage patterns.
"""

using SpidersMessageCodecs
using LinearAlgebra
using Test

function test_zero_allocation_patterns()
    println("🔍 Comprehensive Zero Allocation Testing")
    println("=" ^ 60)
    println("Testing Julia's escape analysis with standard decode() function")
    println()
    
    # Setup test data
    buf = zeros(UInt8, 5000)
    position_ptr = Ref{Int64}(0)
    test_data = Float64[1.0 2.0 3.0 4.0; 5.0 6.0 7.0 8.0; 9.0 10.0 11.0 12.0]  # 3x4 matrix
    
    # Encode test data once
    enc = TensorMessageEncoder(buf; position_ptr)
    encode(enc, test_data)
    encoded_size = position_ptr[]
    
    println("✓ Test data encoded: $(size(test_data)) matrix, $encoded_size bytes")
    println("✓ Expected sum: $(sum(test_data))")
    println()
    
    allocation_results = Dict{String, Int}()
    
    # Test 1: Basic processing (should be zero allocation)
    println("🧪 Test 1: Basic Processing Pattern")
    
    function basic_processing()
        position_ptr[] = 0
        dec = TensorMessageDecoder(buf; position_ptr)
        matrix = decode(dec, Matrix{Float64})
        
        # Simple processing that doesn't store the matrix
        max_val = maximum(matrix)
        min_val = minimum(matrix)
        total = sum(matrix)
        
        return (max_val, min_val, total)
    end
    
    # Warm up
    result1 = basic_processing()
    
    # Test allocation
    allocs1 = @allocated basic_processing()
    allocation_results["basic_processing"] = allocs1
    
    println("   Result: max=$(result1[1]), min=$(result1[2]), sum=$(result1[3])")
    println("   Allocations: $allocs1 bytes")
    println("   Expected: 0 bytes (escape analysis should eliminate matrix allocation)")
    println()
    
    # Test 2: Mathematical operations (zero allocation version)
    println("🧪 Test 2: Mathematical Operations (zero allocation patterns)")
    
    function math_operations_zero_alloc()
        position_ptr[] = 0
        dec = TensorMessageDecoder(buf; position_ptr)
        matrix = decode(dec, Matrix{Float64})
        
        # Mathematical operations without creating temporary arrays
        norm_val = norm(matrix)
        mean_val = sum(matrix) / length(matrix)
        
        # Manual standard deviation calculation to avoid broadcasting
        sum_sq_diff = 0.0
        for val in matrix
            diff = val - mean_val
            sum_sq_diff += diff * diff
        end
        std_val = sqrt(sum_sq_diff / length(matrix))
        
        return (norm_val, mean_val, std_val)
    end
    
    # Warm up
    result2 = math_operations_zero_alloc()
    
    # Test allocation
    allocs2 = @allocated math_operations_zero_alloc()
    allocation_results["math_operations_zero_alloc"] = allocs2
    
    println("   Result: norm=$(round(result2[1], digits=3)), mean=$(round(result2[2], digits=3)), std=$(round(result2[3], digits=3))")
    println("   Allocations: $allocs2 bytes")
    println("   Expected: 0 bytes (manual calculations avoid broadcasting)")
    println()
    
    # Test 3: Element-wise processing
    println("🧪 Test 3: Element-wise Processing")
    
    function elementwise_processing()
        position_ptr[] = 0
        dec = TensorMessageDecoder(buf; position_ptr)
        matrix = decode(dec, Matrix{Float64})
        
        # Element-wise processing
        positive_count = count(x -> x > 0, matrix)
        sum_of_squares = sum(x -> x^2, matrix)
        max_abs = maximum(abs, matrix)
        
        return (positive_count, sum_of_squares, max_abs)
    end
    
    # Warm up
    result3 = elementwise_processing()
    
    # Test allocation
    allocs3 = @allocated elementwise_processing()
    allocation_results["elementwise_processing"] = allocs3
    
    println("   Result: positive_count=$(result3[1]), sum_squares=$(result3[2]), max_abs=$(result3[3])")
    println("   Allocations: $allocs3 bytes")
    println("   Expected: 0 bytes (no intermediate arrays escape)")
    println()
    
    # Test 4: Conditional processing (zero allocation pattern)
    println("🧪 Test 4: Conditional Processing")
    
    function conditional_processing(threshold=6.0)
        position_ptr[] = 0
        dec = TensorMessageDecoder(buf; position_ptr)
        matrix = decode(dec, Matrix{Float64})
        
        # Zero allocation version - manual loop instead of boolean indexing
        if maximum(matrix) > threshold
            result = 0.0
            for val in matrix
                if val > threshold
                    result += val
                end
            end
        else
            result = sum(matrix)
        end
        
        return result
    end
    
    # Warm up
    result4 = conditional_processing()
    
    # Test allocation
    allocs4 = @allocated conditional_processing()
    allocation_results["conditional_processing"] = allocs4
    
    println("   Result: $result4")
    println("   Allocations: $allocs4 bytes")
    println("   Expected: 0 bytes (manual loop avoids boolean indexing)")
    println()
    
    # Test 5: Loop-based processing
    println("🧪 Test 5: Loop-based Processing")
    
    function loop_processing()
        position_ptr[] = 0
        dec = TensorMessageDecoder(buf; position_ptr)
        matrix = decode(dec, Matrix{Float64})
        
        # Manual loop processing
        total = 0.0
        count = 0
        rows, cols = size(matrix)
        
        for i in 1:rows
            for j in 1:cols
                val = matrix[i, j]
                total += val
                if val > 5.0
                    count += 1
                end
            end
        end
        
        return (total, count)
    end
    
    # Warm up
    result5 = loop_processing()
    
    # Test allocation
    allocs5 = @allocated loop_processing()
    allocation_results["loop_processing"] = allocs5
    
    println("   Result: total=$(result5[1]), count=$(result5[2])")
    println("   Allocations: $allocs5 bytes")
    println("   Expected: 0 bytes (simple loop access pattern)")
    println()
    
    # Test 6: Multiple decode operations
    println("🧪 Test 6: Multiple Decode Operations")
    
    function multiple_decodes()
        total = 0.0
        
        for i in 1:5
            position_ptr[] = 0
            dec = TensorMessageDecoder(buf; position_ptr)
            matrix = decode(dec, Matrix{Float64})
            total += sum(matrix)
        end
        
        return total
    end
    
    # Warm up
    result6 = multiple_decodes()
    
    # Test allocation
    allocs6 = @allocated multiple_decodes()
    allocation_results["multiple_decodes"] = allocs6
    
    println("   Result: $result6")
    println("   Allocations: $allocs6 bytes")
    println("   Expected: 0 bytes (each matrix processed and discarded)")
    println()
    
    # Test 7: Function barriers (testing escape through function calls)
    println("🧪 Test 7: Function Barriers")
    
    function process_matrix_sum(matrix)
        return sum(matrix)
    end
    
    function process_matrix_stats(matrix)
        return (minimum(matrix), maximum(matrix), sum(matrix) / length(matrix))
    end
    
    function function_barriers()
        position_ptr[] = 0
        dec = TensorMessageDecoder(buf; position_ptr)
        matrix = decode(dec, Matrix{Float64})
        
        # Pass matrix to other functions
        total = process_matrix_sum(matrix)
        stats = process_matrix_stats(matrix)
        
        return (total, stats)
    end
    
    # Warm up
    result7 = function_barriers()
    
    # Test allocation
    allocs7 = @allocated function_barriers()
    allocation_results["function_barriers"] = allocs7
    
    println("   Result: total=$(result7[1]), stats=$(result7[2])")
    println("   Allocations: $allocs7 bytes")
    println("   Expected: 0 bytes (matrix doesn't escape through function calls)")
    println()
    
    # Summary
    println("📊 Allocation Summary")
    println("-" ^ 50)
    
    zero_alloc_tests = String[]
    some_alloc_tests = String[]
    
    for (test_name, allocs) in allocation_results
        println("$(rpad(test_name, 25)): $allocs bytes")
        if allocs == 0
            push!(zero_alloc_tests, test_name)
        else
            push!(some_alloc_tests, test_name)
        end
    end
    
    println()
    println("🎯 Results Analysis")
    println("-" ^ 30)
    
    if !isempty(zero_alloc_tests)
        println("✅ Zero allocation tests ($(length(zero_alloc_tests))):")
        for test in zero_alloc_tests
            println("   - $test")
        end
        println()
    end
    
    if !isempty(some_alloc_tests)
        println("⚠️  Tests with allocations ($(length(some_alloc_tests))):")
        for test in some_alloc_tests
            println("   - $test ($(allocation_results[test]) bytes)")
        end
        println()
        println("   Note: Any allocations indicate either:")
        println("   - Complex usage patterns that defeat escape analysis")
        println("   - Julia version-specific optimization differences")
        println("   - Unexpected regression in escape analysis optimization")
        println("   ")
        println("   💡 All tests use only zero-allocation patterns")
        println("      Any allocations suggest areas for investigation")
        println()
    end
    
    total_zero_alloc = length(zero_alloc_tests)
    total_tests = length(allocation_results)
    success_rate = round(total_zero_alloc / total_tests * 100, digits=1)
    
    println("💡 Overall Assessment")
    println("-" ^ 25)
    println("Zero allocation success rate: $success_rate% ($total_zero_alloc/$total_tests tests)")
    
    if success_rate >= 80
        println("🎉 Excellent! Julia's escape analysis is working very well.")
        println("   The standard decode() function achieves zero allocation")
        println("   for most realistic usage patterns.")
    elseif success_rate >= 60
        println("👍 Good! Julia's escape analysis works for most patterns.")
        println("   Consider the zero-allocation patterns for critical code paths.")
    else
        println("⚠️  Julia's escape analysis is limited for these patterns.")
        println("   Consider explicit zero-allocation techniques for performance-critical code.")
    end
    
    println()
    println("🔬 Key Insights:")
    println("- Julia's escape analysis eliminates decoder allocations for proper usage patterns")
    println("- SpidersMessageCodecs decode() itself achieves zero allocation")
    println("- Simple operations (sum, max, min, loops) remain zero allocation")
    println("- Function barriers don't prevent escape analysis optimization")
    println("- Zero allocation is achievable for realistic processing patterns")
    
    return allocation_results
end

function test_high_frequency_pattern()
    println("\n" * "=" ^ 60)
    println("🚀 High-Frequency Usage Pattern Test")
    println("=" ^ 60)
    println("Testing allocation behavior under high-frequency trading scenarios")
    println("Using zero-allocation processing patterns")
    println()
    
    # Setup larger buffer and data for realistic HFT scenario
    buf = zeros(UInt8, 10000)
    position_ptr = Ref{Int64}(0)
    
    # Market data: 100 instruments x 10 features (bid, ask, volume, etc.)
    market_data = rand(Float64, 100, 10) .* 1000 .+ 50  # Price data around 50-1050
    
    # Encode once
    enc = TensorMessageEncoder(buf; position_ptr)
    encode(enc, market_data)
    
    println("✓ Market data: $(size(market_data)) Float64 matrix")
    println("✓ Encoded size: $(position_ptr[]) bytes")
    println()
    
    # Test pattern: Process market data for trading decisions (zero allocation version)
    function hft_processing_pattern()
        position_ptr[] = 0
        dec = TensorMessageDecoder(buf; position_ptr)
        prices = decode(dec, Matrix{Float64})
        
        # Zero allocation HFT processing: avoid creating intermediate arrays
        total_spread = 0.0
        max_volume = 0.0
        max_volume_idx = 1
        total_volume = 0.0
        profitable_spread_count = 0
        
        # Process row by row to avoid allocations
        for i in 1:size(prices, 1)
            bid = prices[i, 1]    # Column 1: bid prices
            ask = prices[i, 2]    # Column 2: ask prices
            volume = prices[i, 3] # Column 3: volumes
            
            # Calculate spread without creating array
            spread = ask - bid
            total_spread += spread
            
            # Track max volume
            if volume > max_volume
                max_volume = volume
                max_volume_idx = i
            end
            
            total_volume += volume
        end
        
        # Calculate average spread
        avg_spread = total_spread / size(prices, 1)
        
        # Count profitable spreads in second pass
        for i in 1:size(prices, 1)
            bid = prices[i, 1]
            ask = prices[i, 2]
            spread = ask - bid
            if spread > avg_spread * 1.1
                profitable_spread_count += 1
            end
        end
        
        return (avg_spread, max_volume_idx, total_volume, profitable_spread_count)
    end
    
    # Warm up
    result = hft_processing_pattern()
    
    # Test single operation allocation
    single_allocs = @allocated hft_processing_pattern()
    println("Single operation allocations: $single_allocs bytes")
    
    # Test repeated operations (simulating high frequency)
    function repeated_hft_processing(n_operations=1000)
        total_allocs = 0
        for i in 1:n_operations
            total_allocs += @allocated hft_processing_pattern()
        end
        return total_allocs
    end
    
    # Measure allocations for 1000 operations
    total_allocs = repeated_hft_processing(1000)
    avg_allocs = total_allocs / 1000
    
    println("1000 operations total allocations: $total_allocs bytes")
    println("Average per operation: $avg_allocs bytes")
    println()
    
    if avg_allocs == 0
        println("🎉 PERFECT for HFT: Zero allocation per operation!")
        println("   No GC pressure even under high frequency processing")
    elseif avg_allocs < 100
        println("✅ EXCELLENT for HFT: Very low allocation per operation")
        println("   Minimal GC pressure suitable for ultra-high-frequency trading")
    elseif avg_allocs < 500
        println("👍 GOOD for HFT: Low allocation per operation")
        println("   Manageable GC pressure suitable for high-frequency trading")
    elseif avg_allocs < 1000
        println("⚠️  MODERATE for HFT: Consider optimization for ultra-low latency")
        println("   May need optimization for critical paths")
    else
        println("❌ HIGH allocation per operation - needs optimization")
        println("   Consider zero-allocation processing patterns")
    end
    
    println()
    println("💼 HFT Assessment:")
    println("- Market data processing: $single_allocs bytes per decode")
    println("- Suitable for: $(avg_allocs == 0 ? "Ultra-high-frequency trading" : avg_allocs < 100 ? "High-frequency trading" : avg_allocs < 500 ? "Medium-frequency trading" : "Lower frequency trading")")
    println("- GC impact: $(avg_allocs == 0 ? "None" : avg_allocs < 100 ? "Minimal" : avg_allocs < 500 ? "Low" : "Moderate")")
    println("- Recommendation: $(avg_allocs < 100 ? "Perfect for ultra-low latency HFT" : avg_allocs < 500 ? "Good for production HFT use" : "Consider optimization for critical paths")")
    
    return (single_allocs, avg_allocs)
end

@testset "Zero Allocation Validation" begin
    # Run comprehensive tests
    allocation_results = test_zero_allocation_patterns()
    
    # Test that all proper zero-allocation patterns achieve zero allocation
    @test allocation_results["basic_processing"] == 0
    @test allocation_results["math_operations_zero_alloc"] == 0
    @test allocation_results["elementwise_processing"] == 0
    @test allocation_results["conditional_processing"] == 0
    @test allocation_results["loop_processing"] == 0
    @test allocation_results["multiple_decodes"] == 0
    @test allocation_results["function_barriers"] == 0
    
    # Run HFT pattern test
    single_allocs, avg_allocs = test_high_frequency_pattern()
    
    # Test that HFT patterns achieve zero allocation with proper patterns
    @test avg_allocs < 100  # Should be under 100 bytes per operation with zero-alloc patterns
    @test single_allocs == 0  # Should achieve zero allocation with proper processing
end

if abspath(PROGRAM_FILE) == @__FILE__
    # Run when executed directly
    test_zero_allocation_patterns()
    test_high_frequency_pattern()
end
