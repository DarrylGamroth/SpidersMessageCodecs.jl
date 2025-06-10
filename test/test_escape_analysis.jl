#!/usr/bin/env julia

"""
Test to demonstrate allocation behavior with escaping vs non-escaping decoded values
"""

using SpidersMessageCodecs
using LinearAlgebra

# Global function for testing function call patterns
function inner_function(matrix)
    # Process the array (non-escaping at this level)
    total = 0.0f0
    for val in matrix
        total += val
    end
    return total
end

# Test data setup
function setup_test_data()
    buf = zeros(UInt8, 50000)  # Increased buffer size for larger tests
    position_ptr = Ref{Int64}(0)
    test_data = Float32[1.0 2.0 3.0; 4.0 5.0 6.0]
    
    # Encode test data once
    enc = TensorMessageEncoder(buf; position_ptr)
    encode(enc, test_data)
    
    return buf, position_ptr, test_data
end

# Test 1: Standard decode that ESCAPES (returns the array)
function test_standard_decode_escaping(buf, position_ptr)
    println("🧪 Test 1: Standard decode - ESCAPING (returns array)")
    
    function decode_escaping(buffer, ptr_ref)
        ptr_ref[] = 0
        dec = TensorMessageDecoder(buffer; position_ptr=ptr_ref)
        return decode(dec, Matrix{Float32})  # Array escapes function
    end
    
    # Warm up
    result_escaping = decode_escaping(buf, position_ptr)
    
    # Measure allocations
    allocs_escaping = @allocated decode_escaping(buf, position_ptr)
    println("   Allocations: $allocs_escaping bytes")
    println("   Result size: $(size(result_escaping))")
    println("   Reason: Array must be allocated since it escapes the function")
    println()
    
    return ("escaping_decode", allocs_escaping, "array returned - allocation required")
end
    
# Test 2: Decode that DOESN'T ESCAPE (processes in-place)
function test_standard_decode_non_escaping(buf, position_ptr, test_data)
    println("🧪 Test 2: Standard decode - NON-ESCAPING (processes in-place)")
    
    function decode_non_escaping(buffer, ptr_ref)
        ptr_ref[] = 0
        dec = TensorMessageDecoder(buffer; position_ptr=ptr_ref)
        matrix = decode(dec, Matrix{Float32})  # Decode array
        
        # Process the array without returning it (doesn't escape)
        total = 0.0f0
        for val in matrix
            total += val
        end
        
        return total  # Only scalar escapes, not the array
    end
    
    # Warm up
    result_non_escaping = decode_non_escaping(buf, position_ptr)
    
    # Measure allocations
    allocs_non_escaping = @allocated decode_non_escaping(buf, position_ptr)
    println("   Allocations: $allocs_non_escaping bytes")
    println("   Result (sum): $result_non_escaping")
    println("   Expected: $(sum(test_data))")
    println("   Reason: Compiler may eliminate array allocation via escape analysis")
    println()
    
    return ("non_escaping_decode", allocs_non_escaping, "escape analysis can optimize away")
end

# Test 3: Decode into pre-allocated buffer that DOESN'T ESCAPE
function test_preallocated_buffer_non_escaping(buf, position_ptr, test_data)
    println("🧪 Test 3: Pre-allocated buffer - NON-ESCAPING")
    
    # Pre-allocate outside the measured function
    output_buffer = Matrix{Float32}(undef, 2, 3)
    
    function decode_into_non_escaping(buffer, ptr_ref, out_buf)
        ptr_ref[] = 0
        dec = TensorMessageDecoder(buffer; position_ptr=ptr_ref)
        matrix = decode(dec, Matrix{Float32})
        
        # Copy to pre-allocated buffer without returning decoded array
        copyto!(out_buf, matrix)
        
        # Return just the sum (scalar escape only)
        return sum(out_buf)
    end
    
    # Warm up
    result_buffer = decode_into_non_escaping(buf, position_ptr, output_buffer)
    
    # Measure allocations
    allocs_buffer = @allocated decode_into_non_escaping(buf, position_ptr, output_buffer)
    println("   Allocations: $allocs_buffer bytes")
    println("   Result (sum): $result_buffer")
    println("   Reason: Original array doesn't escape, only copied to retained buffer")
    println()
    
    return ("buffer_copy", allocs_buffer, "original doesn't escape, copied to pre-allocated")
end

# Test 4: Direct processing without intermediate arrays
function test_direct_processing(buf, position_ptr)
    println("🧪 Test 4: Direct processing - NO ARRAYS")
    
    function decode_direct_processing(buffer, ptr_ref)
        ptr_ref[] = 0
        dec = TensorMessageDecoder(buffer; position_ptr=ptr_ref)
        
        # Process elements directly without creating intermediate arrays
        SpidersMessageCodecs.sbe_rewind!(dec)
        SpidersMessageCodecs.skip_dims!(dec)
        SpidersMessageCodecs.skip_origin!(dec)
        
        values_raw = SpidersMessageCodecs.values(dec)
        data_ptr = reinterpret(Ptr{Float32}, pointer(values_raw))
        num_elements = length(values_raw) ÷ sizeof(Float32)
        
        total = 0.0f0
        for i in 1:num_elements
            val = unsafe_load(data_ptr, i)
            total += val
        end
        
        return total
    end
    
    # Warm up
    result_direct = decode_direct_processing(buf, position_ptr)
    
    # Measure allocations
    allocs_direct = @allocated decode_direct_processing(buf, position_ptr)
    println("   Allocations: $allocs_direct bytes")
    println("   Result (sum): $result_direct")
    println("   Reason: No intermediate arrays created at all")
    println()
    
    return ("direct_processing", allocs_direct, "raw pointers, no intermediate arrays")
end

# Test 5: Simple iteration pattern
function test_simple_iteration(buf, position_ptr)
    println("🧪 Test 5: Simple Iteration Pattern")
    
    function simple_iteration(buffer, ptr_ref)
        ptr_ref[] = 0
        dec = TensorMessageDecoder(buffer; position_ptr=ptr_ref)
        matrix = decode(dec, Matrix{Float32})
        
        # Simple iteration - common pattern
        total = 0.0f0
        for val in matrix
            total += val
        end
        
        return total
    end
    
    # Warm up
    result_simple = simple_iteration(buf, position_ptr)
    
    # Measure allocations
    allocs_simple = @allocated simple_iteration(buf, position_ptr)
    println("   Allocations: $allocs_simple bytes")
    println("   Result (sum): $result_simple")
    println("   Reason: Simple iteration over decoded matrix")
    println()
    
    return ("simple_iteration", allocs_simple, "common decode + iterate pattern")
end
    
# Test 6: Dynamic decode - ESCAPING (global capture)
function test_dynamic_decode_escaping_global(buf, position_ptr)
    println("🧪 Test 6: Dynamic decode - ESCAPING (auto type inference) - GLOBAL CAPTURE")
    
    function decode_dynamic_escaping()
        position_ptr[] = 0  # Captures global position_ptr
        dec = TensorMessageDecoder(buf; position_ptr)  # Captures global buf
        return decode(dec)  # No type specified - dynamic inference
    end
    
    # Warm up
    result_dynamic_escaping = decode_dynamic_escaping()
    
    # Measure allocations
    allocs_dynamic_escaping = @allocated decode_dynamic_escaping()
    println("   Allocations: $allocs_dynamic_escaping bytes")
    println("   Result type: $(typeof(result_dynamic_escaping))")
    println("   Result size: $(size(result_dynamic_escaping))")
    println("   Reason: Global variable capture in closure + dynamic type inference")
    println()
    
    return ("dynamic_escaping", allocs_dynamic_escaping, "global capture + type inferred")
end

# Test 7: Dynamic decode - NON-ESCAPING (global capture)
function test_dynamic_decode_non_escaping_global(buf, position_ptr, test_data)
    println("🧪 Test 7: Dynamic decode - NON-ESCAPING (processes in-place) - GLOBAL CAPTURE")
    
    function decode_dynamic_non_escaping()
        position_ptr[] = 0  # Captures global position_ptr
        dec = TensorMessageDecoder(buf; position_ptr)  # Captures global buf
        matrix = decode(dec)  # Dynamic decode
        
        # Process the array without returning it (doesn't escape)
        total = 0.0
        for val in matrix
            total += val
        end
        
        return total  # Only scalar escapes, not the array
    end
    
    # Warm up
    result_dynamic_non_escaping = decode_dynamic_non_escaping()
    
    # Measure allocations
    allocs_dynamic_non_escaping = @allocated decode_dynamic_non_escaping()
    println("   Allocations: $allocs_dynamic_non_escaping bytes")
    println("   Result (sum): $result_dynamic_non_escaping")
    println("   Expected: $(sum(test_data))")
    println("   Reason: Dynamic decode with escape analysis")
    println()
    
    return ("dynamic_non_escaping", allocs_dynamic_non_escaping, "global capture + process in-place")
end

# Test 8: Local variable capture - NON-ESCAPING
function test_local_variable_capture(buf, position_ptr, test_data)
    println("🧪 Test 8: Local variable capture - NON-ESCAPING (compare with Test 7)")
    
    function test_local_capture()
        # Local variables (not global capture)
        local_buf = buf  # Reference to global buf, but as local variable
        local_position_ptr = Ref{Int64}(0)
        
        function inner_decode_local()
            local_position_ptr[] = 0
            dec = TensorMessageDecoder(local_buf; position_ptr=local_position_ptr)
            matrix = decode(dec, Matrix{Float32})
            
            # Process without returning (non-escaping)
            total = 0.0f0
            for val in matrix
                total += val
            end
            
            return total
        end
        
        return inner_decode_local()
    end
    
    # Warm up
    result_local_capture = test_local_capture()
    
    # Measure allocations
    allocs_local_capture = @allocated test_local_capture()
    println("   Allocations: $allocs_local_capture bytes")
    println("   Result (sum): $result_local_capture")
    println("   Expected: $(sum(test_data))")
    println("   Reason: Local variable capture allows better optimization")
    println()
    
    return ("local_capture", allocs_local_capture, "local capture + process in-place")
end

# Test 9: Dynamic decode with type assertion (global capture)
function test_dynamic_with_assertion_global(buf, position_ptr)
    println("🧪 Test 9: Dynamic decode with type assertion - GLOBAL CAPTURE")
    
    function decode_dynamic_assert()
        position_ptr[] = 0  # Captures global position_ptr
        dec = TensorMessageDecoder(buf; position_ptr)  # Captures global buf
        matrix = decode(dec)  # Dynamic decode
        matrix_converted = Matrix{Float32}(matrix)  # Convert to Matrix{Float32}
        
        total = 0.0f0
        for val in matrix_converted
            total += val
        end
        
        return total
    end
    
    # Warm up
    result_dynamic_assert = decode_dynamic_assert()
    
    # Measure allocations
    allocs_dynamic_assert = @allocated decode_dynamic_assert()
    println("   Allocations: $allocs_dynamic_assert bytes")
    println("   Result (sum): $result_dynamic_assert")
    println("   Reason: Global capture + dynamic decode with type assertion for optimization")
    println()
    
    return ("dynamic_with_assert", allocs_dynamic_assert, "global capture + type asserted")
end

# Test 10: No position_ptr - ESCAPING
function test_no_position_ptr_escaping(buf, position_ptr, test_data)
    println("🧪 Test 10: No position_ptr - ESCAPING (returns array)")
    
    # Pre-encode data into buffer (outside measured function)
    position_ptr[] = 0
    enc = TensorMessageEncoder(buf; position_ptr)
    encode(enc, test_data)
    
    function decode_no_ptr_escaping()
        # Decode without position_ptr - uses internal position tracking
        dec = TensorMessageDecoder(buf)
        return decode(dec, Matrix{Float32})  # Array escapes function
    end
    
    # Warm up
    result_no_ptr_escaping = decode_no_ptr_escaping()
    
    # Measure allocations
    allocs_no_ptr_escaping = @allocated decode_no_ptr_escaping()
    println("   Allocations: $allocs_no_ptr_escaping bytes")
    println("   Result size: $(size(result_no_ptr_escaping))")
    println("   Reason: No position_ptr parameter - simpler API, no buffer overhead")
    println()
    
    return ("no_ptr_escaping", allocs_no_ptr_escaping, "no position_ptr, fair comparison")
end

# Test 11: No position_ptr - NON-ESCAPING
function test_no_position_ptr_non_escaping(buf, position_ptr, test_data)
    println("🧪 Test 11: No position_ptr - NON-ESCAPING (processes in-place)")
    
    # Pre-encode data into buffer (outside measured function)
    position_ptr[] = 0
    enc = TensorMessageEncoder(buf; position_ptr)
    encode(enc, test_data)
    
    function decode_no_ptr_non_escaping(buffer)
        # Decode without position_ptr
        dec = TensorMessageDecoder(buffer)
        matrix = decode(dec, Matrix{Float32})
        
        # Process the array without returning it (doesn't escape)
        total = 0.0f0
        for val in matrix
            total += val
        end
        
        return total  # Only scalar escapes, not the array
    end
    
    # Warm up
    result_no_ptr_non_escaping = decode_no_ptr_non_escaping(buf)
    
    # Measure allocations
    allocs_no_ptr_non_escaping = @allocated decode_no_ptr_non_escaping(buf)
    println("   Allocations: $allocs_no_ptr_non_escaping bytes")
    println("   Result (sum): $result_no_ptr_non_escaping")
    println("   Expected: $(sum(test_data))")
    println("   Reason: No position_ptr - escape analysis optimization test")
    println()
    
    return ("no_ptr_non_escaping", allocs_no_ptr_non_escaping, "fair test - no buffer allocation")
end

# Test 12: Dynamic no position_ptr - ESCAPING
function test_dynamic_no_position_ptr_escaping(buf, position_ptr, test_data)
    println("🧪 Test 12: Dynamic no position_ptr - ESCAPING (auto type inference)")
    
    # Pre-encode data into buffer (outside measured function)
    position_ptr[] = 0
    enc = TensorMessageEncoder(buf; position_ptr)
    encode(enc, test_data)
    
    function decode_dynamic_no_ptr_escaping(buffer)
        # Decode without position_ptr - dynamic
        dec = TensorMessageDecoder(buffer)
        return decode(dec)  # No type specified - dynamic inference
    end
    
    # Warm up
    result_dynamic_no_ptr_escaping = decode_dynamic_no_ptr_escaping(buf)
    
    # Measure allocations
    allocs_dynamic_no_ptr_escaping = @allocated decode_dynamic_no_ptr_escaping(buf)
    println("   Allocations: $allocs_dynamic_no_ptr_escaping bytes")
    println("   Result type: $(typeof(result_dynamic_no_ptr_escaping))")
    println("   Result size: $(size(result_dynamic_no_ptr_escaping))")
    println("   Reason: Dynamic decode without position_ptr - fair test")
    println()
    
    return ("dynamic_no_ptr_escaping", allocs_dynamic_no_ptr_escaping, "fair test - no buffer allocation")
end

# Test 13: Dynamic no position_ptr - NON-ESCAPING
function test_dynamic_no_position_ptr_non_escaping(buf, position_ptr, test_data)
    println("🧪 Test 13: Dynamic no position_ptr - NON-ESCAPING (processes in-place)")
    
    # Pre-encode data into buffer (outside measured function)
    position_ptr[] = 0
    enc = TensorMessageEncoder(buf; position_ptr)
    encode(enc, test_data)
    
    function decode_dynamic_no_ptr_non_escaping(buffer)
        # Decode without position_ptr - dynamic
        dec = TensorMessageDecoder(buffer)
        matrix = decode(dec)  # Dynamic decode
        
        # Process the array without returning it (doesn't escape)
        total = 0.0
        for val in matrix
            total += val
        end
        
        return total  # Only scalar escapes, not the array
    end
    
    # Warm up
    result_dynamic_no_ptr_non_escaping = decode_dynamic_no_ptr_non_escaping(buf)
    
    # Measure allocations
    allocs_dynamic_no_ptr_non_escaping = @allocated decode_dynamic_no_ptr_non_escaping(buf)
    println("   Allocations: $allocs_dynamic_no_ptr_non_escaping bytes")
    println("   Result (sum): $result_dynamic_no_ptr_non_escaping")
    println("   Expected: $(sum(test_data))")
    println("   Reason: Dynamic decode without position_ptr - fair test")
    println()
    
    return ("dynamic_no_ptr_non_escaping", allocs_dynamic_no_ptr_non_escaping, "fair test - no buffer allocation")
end

# Test 14: Multiple buffer reuse - ESCAPING
function test_multiple_buffer_reuse_escaping(buf, position_ptr, test_data)
    println("🧪 Test 14: Multiple buffer reuse - ESCAPING (realistic pattern)")
    
    # Pre-encode data into buffer (outside measured function)
    position_ptr[] = 0
    enc = TensorMessageEncoder(buf; position_ptr)
    encode(enc, test_data)
    
    function decode_multiple_times_escaping(buffer, ptr_ref)
        # Realistic pattern: decode the same buffer multiple times
        results = Matrix{Float32}[]
        
        for _ in 1:3  # Decode 3 times 
            ptr_ref[] = 0
            dec = TensorMessageDecoder(buffer; position_ptr=ptr_ref)
            matrix = decode(dec, Matrix{Float32})
            push!(results, matrix)  # Arrays escape
        end
        
        return results
    end
    
    # Warm up
    result_multiple = decode_multiple_times_escaping(buf, position_ptr)
    
    # Measure allocations
    allocs_multiple = @allocated decode_multiple_times_escaping(buf, position_ptr)
    println("   Allocations: $allocs_multiple bytes")
    println("   Results count: $(length(result_multiple))")
    println("   Reason: Multiple decodes from same buffer - realistic usage pattern")
    println()
    
    return ("buffer_reuse_escaping", allocs_multiple, "multiple decodes with shared buffer")
end

# Test 15: Large vs small array size comparison
function test_large_vs_small_array_escaping(buf, position_ptr)
    println("🧪 Test 15: Large vs Small Array Size Comparison - ESCAPING")
    
    # Small array (2x3 = 6 elements)
    small_data = Float32[1.0 2.0 3.0; 4.0 5.0 6.0]
    position_ptr[] = 0
    enc = TensorMessageEncoder(buf; position_ptr)
    encode(enc, small_data)
    
    function decode_small_escaping(buffer, ptr_ref)
        ptr_ref[] = 0
        dec = TensorMessageDecoder(buffer; position_ptr=ptr_ref)
        return decode(dec, Matrix{Float32})
    end
    
    # Warm up and measure small array
    _ = decode_small_escaping(buf, position_ptr)
    allocs_decode_small = @allocated decode_small_escaping(buf, position_ptr)
    println("   Small array (2x3) decode allocations: $allocs_decode_small bytes")
    
    # Medium array (20x20 = 400 elements) - reasonable size for testing
    medium_data = Float32[i*j*0.01 for i in 1:20, j in 1:20]
    position_ptr[] = 0
    encode(enc, medium_data)
    
    function decode_medium_escaping(buffer, ptr_ref)
        ptr_ref[] = 0
        dec = TensorMessageDecoder(buffer; position_ptr=ptr_ref)
        return decode(dec, Matrix{Float32})
    end
    
    # Warm up and measure medium array
    _ = decode_medium_escaping(buf, position_ptr)
    allocs_decode_medium = @allocated decode_medium_escaping(buf, position_ptr)
    println("   Medium array (20x20) decode allocations: $allocs_decode_medium bytes")
    
    # Calculate scaling
    if allocs_decode_small > 0
        scaling_factor = allocs_decode_medium / allocs_decode_small
    else
        scaling_factor = allocs_decode_medium > 0 ? "infinite" : "both zero"
    end
    
    println("   Size scaling factor: $scaling_factor")
    println("   Reason: Compare allocation scaling with array size")
    println()
    
    return ("large_vs_small", allocs_decode_medium - allocs_decode_small, "allocation scaling with size")
end

# Test 15b: Large 500x500 array test
function test_large_array_500x500(buf, position_ptr, test_data)
    println("🧪 Test 15b: Large 500x500 Array - Allocation Analysis")
    
    # Large array (500x500 = 250,000 elements) - significant size
    large_data = Float32[sin(i*0.01) * cos(j*0.01) for i in 1:500, j in 1:500]
    println("   Generated 500x500 array ($(length(large_data)) elements, $(sizeof(large_data)) bytes data)")
    
    # Ensure we have enough buffer space
    large_buf = zeros(UInt8, 2_000_000)  # 2MB buffer for large array
    large_position_ptr = Ref{Int64}(0)
    
    # Encode the large array
    large_position_ptr[] = 0
    enc = TensorMessageEncoder(large_buf; position_ptr=large_position_ptr)
    encode(enc, large_data)
    encoded_size = large_position_ptr[]
    println("   Encoded size: $encoded_size bytes")
    
    # Test 1: Type-stable decode (baseline)
    function decode_large_stable(buffer, ptr_ref)
        ptr_ref[] = 0
        dec = TensorMessageDecoder(buffer; position_ptr=ptr_ref)
        return decode(dec, Matrix{Float32})  # Explicit type
    end
    
    # Test 2: Type-unstable decode (compare with Test 16)
    function decode_large_unstable(buffer, ptr_ref)
        ptr_ref[] = 0
        dec = TensorMessageDecoder(buffer; position_ptr=ptr_ref)
        matrix = decode(dec)  # Dynamic decode - type unstable
        return sum(matrix)  # Force processing to see allocation impact
    end
    
    # Warm up both functions
    _ = decode_large_stable(large_buf, large_position_ptr)
    _ = decode_large_unstable(large_buf, large_position_ptr)
    
    # Measure allocations
    allocs_stable = @allocated decode_large_stable(large_buf, large_position_ptr)
    allocs_unstable = @allocated decode_large_unstable(large_buf, large_position_ptr)
    
    println("   Large array stable decode:   $allocs_stable bytes")
    println("   Large array unstable decode: $allocs_unstable bytes")
    println("   Type instability penalty:    $(allocs_unstable - allocs_stable) bytes")
    
    # Compare with small array type instability (from Test 16)
    # Re-run Test 16 pattern for comparison
    position_ptr[] = 0
    enc_small = TensorMessageEncoder(buf; position_ptr)
    encode(enc_small, test_data)
    
    function small_unstable()
        position_ptr[] = 0
        dec = TensorMessageDecoder(buf; position_ptr)
        matrix = decode(dec)  # Dynamic decode
        return sum(matrix)
    end
    
    _ = small_unstable()
    allocs_small_unstable = @allocated small_unstable()
    
    # Analysis
    if allocs_small_unstable > 0
        penalty_ratio = (allocs_unstable - allocs_stable) / allocs_small_unstable
        println("   Type instability penalty ratio (large/small): $(round(penalty_ratio, digits=2))x")
    end
    
    data_ratio = length(large_data) / length(test_data)
    println("   Data size ratio (large/small): $(round(data_ratio, digits=0))x")
    println("   Reason: Does type instability penalty scale with array size?")
    println()
    
    return ("large_500x500", allocs_unstable - allocs_stable, "500x500 array type instability penalty")
end

# Test 16: Type-unstable return type
function test_type_unstable_return(buf, position_ptr, test_data)
    println("🧪 Test 16: Type-unstable Return Type")
    
    function unstable_return_type()
        position_ptr[] = 0
        dec = TensorMessageDecoder(buf; position_ptr)
        matrix = decode(dec)  # Dynamic decode - type unstable
        
        # Process the array (escaping)
        total = 0.0f0
        for val in matrix
            total += val
        end
        
        return total
    end
    
    # Warm up
    result_unstable = unstable_return_type()
    
    # Measure allocations
    allocs_unstable = @allocated unstable_return_type()
    println("   Allocations: $allocs_unstable bytes")
    println("   Result (sum): $result_unstable")
    println("   Reason: Type-unstable return type causes allocation")
    println()
    
    return ("type_unstable", allocs_unstable, "dynamic decode - type unstable")
end

# Test 17: Nested function calls
function test_nested_function_calls(buf, position_ptr, test_data)
    println("🧪 Test 17: Nested Function Calls")
    
    function outer_function()
        position_ptr[] = 0
        dec = TensorMessageDecoder(buf; position_ptr)
        matrix = decode(dec, Matrix{Float32})
        
        return inner_function(matrix)  # Uses global inner_function
    end
    
    # Warm up
    result_nested = outer_function()
    
    # Measure allocations
    allocs_nested = @allocated outer_function()
    println("   Allocations: $allocs_nested bytes")
    println("   Result (sum): $result_nested")
    println("   Reason: Nested function calls - outer escapes")
    println()
    
    return ("nested_calls", allocs_nested, "outer function escapes")
end

# Test 17b: Global vs Anonymous Function Call Comparison  
function test_global_vs_anonymous_function_calls(buf, position_ptr, test_data)
    println("🧪 Test 17b: Global vs Anonymous Function Call Comparison")
    
    # Test A: Call global function (forces escape - function could be redefined)
    function call_global_function()
        position_ptr[] = 0
        dec = TensorMessageDecoder(buf; position_ptr)
        matrix = decode(dec, Matrix{Float32})
        
        return inner_function(matrix)  # Call to GLOBAL function - uncertain identity
    end
    
    # Test B: Call anonymous/nested function (compiler knows the identity)
    function call_anonymous_function()
        position_ptr[] = 0
        dec = TensorMessageDecoder(buf; position_ptr)
        matrix = decode(dec, Matrix{Float32})
        
        # Define local anonymous function - compiler knows this can't change
        local_processor = function(arr)
            total = 0.0f0
            for val in arr
                total += val
            end
            return total
        end
        
        return local_processor(matrix)  # Call to LOCAL function - known identity
    end
    
    # Test C: Built-in function (most optimizable)
    function call_builtin_function()
        position_ptr[] = 0
        dec = TensorMessageDecoder(buf; position_ptr)
        matrix = decode(dec, Matrix{Float32})
        
        return sum(matrix)  # Built-in function - compiler can inline/optimize
    end
    
    # Test D: Inline processing (definitely no escape)
    function inline_processing()
        position_ptr[] = 0
        dec = TensorMessageDecoder(buf; position_ptr)
        matrix = decode(dec, Matrix{Float32})
        
        # Inline the exact same logic
        total = 0.0f0
        for val in matrix
            total += val
        end
        return total
    end
    
    # Warm up all functions
    result_global = call_global_function()
    result_anonymous = call_anonymous_function()
    result_builtin = call_builtin_function()
    result_inline = inline_processing()
    
    # Measure allocations
    allocs_global = @allocated call_global_function()
    allocs_anonymous = @allocated call_anonymous_function()
    allocs_builtin = @allocated call_builtin_function()
    allocs_inline = @allocated inline_processing()
    
    println("   Global function call:    $allocs_global bytes")
    println("   Anonymous function call: $allocs_anonymous bytes") 
    println("   Built-in function call:  $allocs_builtin bytes")
    println("   Inline processing:       $allocs_inline bytes")
    println("   Results: $result_global, $result_anonymous, $result_builtin, $result_inline")
    println()
    println("   🔍 Analysis:")
    println("      Global vs Anonymous:    $(allocs_global - allocs_anonymous) bytes difference")
    println("      Anonymous vs Built-in:  $(allocs_anonymous - allocs_builtin) bytes difference") 
    println("      Built-in vs Inline:     $(allocs_builtin - allocs_inline) bytes difference")
    println()
    println("   Reason: Global functions force escape because compiler can't prove function identity")
    println("          Anonymous/local functions have known identity and can be optimized")
    println("          Built-in functions are most optimizable, inline is guaranteed optimal")
    println()
    
    return ("global_vs_anonymous", allocs_global - allocs_anonymous, "function identity affects escape analysis")
end

# Test 18: Exception handling (try-catch)
function test_exception_handling(buf, position_ptr, test_data)
    println("🧪 Test 18: Exception Handling (try-catch)")
    
    function decode_with_exception_handling()
        try
            position_ptr[] = 0
            dec = TensorMessageDecoder(buf; position_ptr)
            matrix = decode(dec, Matrix{Float32})
            
            # Force an exception for testing (out of bounds)
            return matrix[3, 3]
        catch e
            return -1.0f0  # Return a default value on exception
        end
    end
    
    # Warm up
    result_exception = decode_with_exception_handling()
    
    # Measure allocations
    allocs_exception = @allocated decode_with_exception_handling()
    println("   Allocations: $allocs_exception bytes")
    println("   Result (with exception): $result_exception")
    println("   Reason: Exception handling adds overhead")
    println()
    
    return ("exception_handling", allocs_exception, "try-catch overhead")
end

# Test 19: Multiple array operations
function test_multiple_array_operations(buf, position_ptr, test_data)
    println("🧪 Test 19: Multiple Array Operations")
    
    function decode_multiple_operations()
        position_ptr[] = 0
        dec = TensorMessageDecoder(buf; position_ptr)
        matrix = decode(dec, Matrix{Float32})
        
        # Multiple operations - may optimize better
        result = sum(matrix) * maximum(matrix) - minimum(matrix)
        
        return result
    end
    
    # Warm up
    result_operations = decode_multiple_operations()
    
    # Measure allocations
    allocs_operations = @allocated decode_multiple_operations()
    println("   Allocations: $allocs_operations bytes")
    println("   Result (with operations): $result_operations")
    println("   Reason: Multiple operations - potential optimization")
    println()
    
    return ("multi_operations", allocs_operations, "multiple ops - sum, min, max")
end

# Test 20: Conditional data-dependent processing
function test_conditional_processing(buf, position_ptr, test_data)
    println("🧪 Test 20: Conditional Data-Dependent Processing")
    
    function decode_conditional_processing()
        position_ptr[] = 0
        dec = TensorMessageDecoder(buf; position_ptr)
        matrix = decode(dec, Matrix{Float32})
        
        # Conditional processing based on data
        total = 0.0f0
        for val in matrix
            if val > 0
                total += val
            else
                total -= val
            end
        end
        
        return total
    end
    
    # Warm up
    result_conditional = decode_conditional_processing()
    
    # Measure allocations
    allocs_conditional = @allocated decode_conditional_processing()
    println("   Allocations: $allocs_conditional bytes")
    println("   Result (conditional processing): $result_conditional")
    println("   Reason: Conditional processing - data-dependent")
    println()
    
    return ("conditional", allocs_conditional, "conditional processing based on data")
end

# Summary and analysis
function generate_summary_and_analysis(results, allocs_escaping, allocs_non_escaping)
    println("📊 Summary of Allocation Behavior")
    println("-" ^ 70)
    
    # Create a mapping for display names with clearer descriptions
    display_names = Dict(
        "escaping_decode" => "Typed decode (returns array)",
        "non_escaping_decode" => "Typed decode (process in-place)", 
        "buffer_copy" => "Copy to pre-allocated buffer",
        "direct_processing" => "Raw pointer processing",
        "simple_iteration" => "Basic iteration pattern",
        "dynamic_escaping" => "Dynamic decode (returns array, global vars)",
        "dynamic_non_escaping" => "Dynamic decode (process in-place, global vars)",
        "local_capture" => "Dynamic decode (process in-place, local vars)",
        "dynamic_with_assert" => "Dynamic decode with type assertion (global vars)",
        "no_ptr_escaping" => "No position_ptr (returns array)",
        "no_ptr_non_escaping" => "No position_ptr (process in-place)",
        "dynamic_no_ptr_escaping" => "No position_ptr + dynamic (returns array)",
        "dynamic_no_ptr_non_escaping" => "No position_ptr + dynamic (process in-place)",
        "buffer_reuse_escaping" => "Multiple buffer reuse (realistic pattern)",
        "large_vs_small" => "Large vs small array size comparison",
        "large_500x500" => "Large 500x500 array (type instability scaling)",
        "type_unstable" => "Type-unstable return type",
        "nested_calls" => "Nested function calls",
        "global_vs_anonymous" => "Global vs anonymous function calls",
        "exception_handling" => "Exception handling (try-catch)",
        "multi_operations" => "Complex array operations",
        "conditional" => "Conditional data-dependent processing",
        "type_stability_comparison" => "Stable vs unstable decode comparison",
        "type_inference_failure" => "Dynamic type selection failure",
        "multiple_dynamic_calls" => "Multiple unstable calls (compounding)",
        "return_type_annotation" => "Return type annotation effects",
        "code_analysis" => "@code_warntype analysis patterns"
    )
    
    # Find the maximum width needed for display names and allocations
    max_name_width = maximum(length(get(display_names, result[1], result[1])) for result in results)
    max_alloc_width = maximum(length(string(result[2])) for result in results)
    
    # Print header
    name_col_width = max(max_name_width + 1, 45)  # Minimum 45 chars for readability
    alloc_col_width = max(max_alloc_width + 6, 10)  # +6 for " bytes"
    
    println("$(rpad("Test Pattern", name_col_width)) $(rpad("Allocations", alloc_col_width)) Description")
    println("$(repeat("-", name_col_width)) $(repeat("-", alloc_col_width)) $(repeat("-", 30))")
    
    # Print each result with aligned columns
    for (test_name, allocs, description) in results
        display_name = get(display_names, test_name, test_name)
        alloc_str = string(allocs) * " bytes"
        println("$(rpad(display_name, name_col_width)) $(rpad(alloc_str, alloc_col_width)) $description")
    end
    
    println()
    
    # Analysis
    if allocs_non_escaping == 0
        println("✅ Julia's escape analysis eliminated the allocation!")
        println("   The decoded array was optimized away since it didn't escape")
    elseif allocs_non_escaping < allocs_escaping
        println("⚡ Partial optimization: $(allocs_escaping - allocs_non_escaping) bytes saved")
        println("   Some allocation reduction due to escape analysis")
    else
        println("⚠️  No escape analysis optimization detected")
        println("   This may depend on Julia version and compilation flags")
    end
    
    println()
    println("💡 Key Lessons:")
    println("   1. ENSURE TYPE STABILITY - Use explicit types: decode(dec, Matrix{Float32})")
    println("      Dynamic typing (decode(dec)) adds allocation overhead")
    println()
    println("   2. KEEP FUNCTIONS SMALL - Avoid global variable capture in closures")
    println("      Small functions with parameters optimize better than closures")
    println()
    println("   3. REUSE BUFFERS - position_ptr pattern saves ~2000 bytes per operation")
    println("      Without buffer reuse, each decode allocates a new buffer")
    
    return results
end

# Utility functions for result analysis
function filter_results_by_pattern(results, pattern_regex)
    """Filter results by test name pattern"""
    return filter(result -> occursin(pattern_regex, result[1]), results)
end

function get_allocation_stats(results)
    """Get basic statistics about allocations across all tests"""
    allocations = [result[2] for result in results]
    
    return (
        min_allocs = minimum(allocations),
        max_allocs = maximum(allocations),
        mean_allocs = round(sum(allocations) / length(allocations), digits=1),
        zero_alloc_count = count(==(0), allocations),
        total_tests = length(allocations)
    )
end

# Enhanced summary function with focused lessons
function generate_enhanced_summary_and_analysis(results, allocs_escaping, allocs_non_escaping)
    generate_summary_and_analysis(results, allocs_escaping, allocs_non_escaping)
    return results
end

# Specialized test runners for focused analysis
function run_test_category(category_name, test_functions, data_setup_fn=setup_test_data)
    """Generic test runner for any category of tests"""
    println("🔬 $(category_name)")
    println("=" ^ length(category_name))
    
    setup_data = data_setup_fn()
    if length(setup_data) == 3
        buf, position_ptr, test_data = setup_data
        println("✓ Test data encoded: $(size(test_data)) matrix")
    else
        test_data = setup_data
        println("✓ Test data prepared")
    end
    println()
    
    results = []
    for test_fn in test_functions
        if length(setup_data) == 3
            buf, position_ptr, test_data = setup_data
            if hasmethod(test_fn, (typeof(buf), typeof(position_ptr), typeof(test_data)))
                push!(results, test_fn(buf, position_ptr, test_data))
            elseif hasmethod(test_fn, (typeof(buf), typeof(position_ptr)))
                push!(results, test_fn(buf, position_ptr))
            else
                push!(results, test_fn(test_data))
            end
        else
            push!(results, test_fn(setup_data))
        end
    end
    
    return results
end

function test_minimal_core_only()
    """Run only the most basic tests (Tests 1-2) for quick validation"""
    println("⚡ Minimal Core Tests")
    println("=" ^ 20)
    
    buf, position_ptr, test_data = setup_test_data()
    println("✓ Test data encoded: $(size(test_data)) matrix")
    println()
    
    results = []
    push!(results, test_standard_decode_escaping(buf, position_ptr))
    push!(results, test_standard_decode_non_escaping(buf, position_ptr, test_data))
    
    # Extract values for analysis
    allocs_escaping = results[1][2]
    allocs_non_escaping = results[2][2]
    
    return generate_summary_and_analysis(results, allocs_escaping, allocs_non_escaping)
end

function test_optimization_comparison()
    """Compare different optimization strategies side by side"""
    println("🎯 Optimization Strategy Comparison")
    println("=" ^ 35)
    
    buf, position_ptr, test_data = setup_test_data()
    println("✓ Test data encoded: $(size(test_data)) matrix")
    println()
    
    results = []
    # Basic patterns
    push!(results, test_standard_decode_non_escaping(buf, position_ptr, test_data))
    push!(results, test_preallocated_buffer_non_escaping(buf, position_ptr, test_data))
    push!(results, test_direct_processing(buf, position_ptr))
    push!(results, test_simple_iteration(buf, position_ptr))
    
    # Closure patterns  
    push!(results, test_dynamic_decode_non_escaping_global(buf, position_ptr, test_data))
    push!(results, test_local_variable_capture(buf, position_ptr, test_data))
    
    # Find best performing
    best_result = minimum(results, by=x->x[2])
    worst_result = maximum(results, by=x->x[2])
    
    println()
    println("🏆 Best performing pattern:")
    println("   $(best_result[1]): $(best_result[2]) bytes - $(best_result[3])")
    println()
    println("⚠️  Worst performing pattern:")
    println("   $(worst_result[1]): $(worst_result[2]) bytes - $(worst_result[3])")
    
    return generate_summary_and_analysis(results, worst_result[2], best_result[2])
end

function test_closure_impact_analysis()
    """Detailed analysis of closure capture impact on allocations"""
    println("🔍 Closure Capture Impact Analysis")
    println("=" ^ 35)
    
    buf, position_ptr, test_data = setup_test_data()
    println("✓ Test data encoded: $(size(test_data)) matrix")
    println()
    
    # Test global vs local capture patterns
    results = []
    push!(results, test_dynamic_decode_escaping_global(buf, position_ptr))
    push!(results, test_dynamic_decode_non_escaping_global(buf, position_ptr, test_data))
    push!(results, test_local_variable_capture(buf, position_ptr, test_data))
    push!(results, test_dynamic_with_assertion_global(buf, position_ptr))
    
    # Calculate closure overhead
    global_non_escaping = results[2][2]  # Global capture non-escaping
    local_capture = results[3][2]        # Local capture
    closure_overhead = global_non_escaping - local_capture
    
    println()
    println("📈 Closure Capture Overhead Analysis:")
    println("   Global capture (non-escaping): $(global_non_escaping) bytes")
    println("   Local capture (non-escaping):  $(local_capture) bytes")
    println("   Closure overhead:              $(closure_overhead) bytes")
    if global_non_escaping > 0
        println("   Overhead percentage:           $(round(closure_overhead/global_non_escaping*100, digits=1))%")
    end
    
    return generate_summary_and_analysis(results, global_non_escaping, local_capture)
end

# Helper functions for running specific test categories

function run_core_escape_tests(buf, position_ptr, test_data)
    """Run core escape analysis tests (Tests 1-5)"""
    results = []
    push!(results, test_standard_decode_escaping(buf, position_ptr))
    push!(results, test_standard_decode_non_escaping(buf, position_ptr, test_data))
    push!(results, test_preallocated_buffer_non_escaping(buf, position_ptr, test_data))
    push!(results, test_direct_processing(buf, position_ptr))
    push!(results, test_simple_iteration(buf, position_ptr))
    return results
end

function run_closure_capture_tests(buf, position_ptr, test_data)
    """Run closure capture tests (Tests 6-9)"""
    results = []
    push!(results, test_dynamic_decode_escaping_global(buf, position_ptr))
    push!(results, test_dynamic_decode_non_escaping_global(buf, position_ptr, test_data))
    push!(results, test_local_variable_capture(buf, position_ptr, test_data))
    push!(results, test_dynamic_with_assertion_global(buf, position_ptr))
    return results
end

function run_no_position_ptr_tests(buf, position_ptr, test_data)
    """Run no position_ptr tests (Tests 10-13)"""
    results = []
    push!(results, test_no_position_ptr_escaping(buf, position_ptr, test_data))
    push!(results, test_no_position_ptr_non_escaping(buf, position_ptr, test_data))
    push!(results, test_dynamic_no_position_ptr_escaping(buf, position_ptr, test_data))
    push!(results, test_dynamic_no_position_ptr_non_escaping(buf, position_ptr, test_data))
    return results
end

function run_advanced_tests(buf, position_ptr, test_data)
    """Run advanced allocation tests (Tests 14-20)"""
    results = []
    push!(results, test_multiple_buffer_reuse_escaping(buf, position_ptr, test_data))
    push!(results, test_large_vs_small_array_escaping(buf, position_ptr))
    push!(results, test_large_array_500x500(buf, position_ptr, test_data))  # Test 15b
    push!(results, test_type_unstable_return(buf, position_ptr, test_data))
    push!(results, test_nested_function_calls(buf, position_ptr, test_data))
    push!(results, test_global_vs_anonymous_function_calls(buf, position_ptr, test_data))  # Test 17b
    push!(results, test_exception_handling(buf, position_ptr, test_data))
    push!(results, test_multiple_array_operations(buf, position_ptr, test_data))
    push!(results, test_conditional_processing(buf, position_ptr, test_data))
    return results
end

function test_core_escape_analysis_only()
    """Run only the core escape analysis tests (Tests 1-5)"""
    println("🔬 Core Escape Analysis Tests")
    println("=" ^ 30)
    
    buf, position_ptr, test_data = setup_test_data()
    println("✓ Test data encoded: $(size(test_data)) matrix")
    println()
    
    results = run_core_escape_tests(buf, position_ptr, test_data)
    
    # Extract values for analysis
    allocs_escaping = results[1][2]
    allocs_non_escaping = results[2][2]
    
    return generate_summary_and_analysis(results, allocs_escaping, allocs_non_escaping)
end

function test_closure_capture_only()
    """Run only the closure capture tests (Tests 6-9)"""
    println("🔬 Closure Capture Analysis Tests")
    println("=" ^ 35)
    
    buf, position_ptr, test_data = setup_test_data()
    println("✓ Test data encoded: $(size(test_data)) matrix")
    println()
    
    results = run_closure_capture_tests(buf, position_ptr, test_data)
    
    # Calculate closure overhead
    global_non_escaping = results[2][2]  # Global capture non-escaping
    local_capture = results[3][2]        # Local capture
    closure_overhead = global_non_escaping - local_capture
    
    println()
    println("📈 Closure Capture Overhead Analysis:")
    println("   Global capture (non-escaping): $(global_non_escaping) bytes")
    println("   Local capture (non-escaping):  $(local_capture) bytes")
    println("   Closure overhead:              $(closure_overhead) bytes")
    if global_non_escaping > 0
        println("   Overhead percentage:           $(round(closure_overhead/global_non_escaping*100, digits=1))%")
    end
    
    return generate_summary_and_analysis(results, global_non_escaping, local_capture)
end

function test_buffer_allocation_only()
    """Run only the buffer allocation tests (Tests 10-13)"""
    println("🔬 Buffer Allocation Tests")
    println("=" ^ 25)
    
    buf, position_ptr, test_data = setup_test_data()
    println("✓ Test data encoded: $(size(test_data)) matrix")
    println()
    
    results = run_no_position_ptr_tests(buf, position_ptr, test_data)
    
    # Extract values for analysis
    allocs_escaping = results[1][2]  # No ptr escaping
    allocs_non_escaping = results[2][2]  # No ptr non-escaping
    
    return generate_summary_and_analysis(results, allocs_escaping, allocs_non_escaping)
end

function test_advanced_allocation_patterns()
    """Run only the advanced allocation pattern tests (Tests 14-20)"""
    println("🔬 Advanced Allocation Pattern Tests")
    println("=" ^ 40)
    
    buf, position_ptr, test_data = setup_test_data()
    println("✓ Test data encoded: $(size(test_data)) matrix")
    println()
    
    results = run_advanced_tests(buf, position_ptr, test_data)
    
    # Find most interesting patterns using manual comparison
    max_allocs = results[1]
    min_allocs = results[1]
    for result in results
        if result[2] > max_allocs[2]
            max_allocs = result
        end
        if result[2] < min_allocs[2]
            min_allocs = result
        end
    end
    
    println()
    println("📈 Advanced Pattern Analysis:")
    println("   Highest allocations: $(max_allocs[1]) - $(max_allocs[2]) bytes")
    println("   Lowest allocations:  $(min_allocs[1]) - $(min_allocs[2]) bytes")
    println("   Pattern difference:  $(max_allocs[2] - min_allocs[2]) bytes")
    
    return generate_summary_and_analysis(results, max_allocs[2], min_allocs[2])
end

# Main test function that orchestrates all tests
function test_escaping_vs_non_escaping()
    println("🔍 Testing Allocation Behavior: Escaping vs Non-Escaping")
    println("=" ^ 60)
    
    # Setup test data
    buf, position_ptr, test_data = setup_test_data()
    
    println("✓ Test data encoded: $(size(test_data)) matrix")
    println()
    
    # Run all individual tests and collect results
    results = []
    
    # Core escape analysis tests (Tests 1-5)
    push!(results, test_standard_decode_escaping(buf, position_ptr))
    push!(results, test_standard_decode_non_escaping(buf, position_ptr, test_data))
    push!(results, test_preallocated_buffer_non_escaping(buf, position_ptr, test_data))
    push!(results, test_direct_processing(buf, position_ptr))
    push!(results, test_simple_iteration(buf, position_ptr))
    
    # Closure capture tests (Tests 6-9)
    push!(results, test_dynamic_decode_escaping_global(buf, position_ptr))
    push!(results, test_dynamic_decode_non_escaping_global(buf, position_ptr, test_data))
    push!(results, test_local_variable_capture(buf, position_ptr, test_data))
    push!(results, test_dynamic_with_assertion_global(buf, position_ptr))
    
    # No position_ptr tests (Tests 10-13)
    push!(results, test_no_position_ptr_escaping(buf, position_ptr, test_data))
    push!(results, test_no_position_ptr_non_escaping(buf, position_ptr, test_data))
    push!(results, test_dynamic_no_position_ptr_escaping(buf, position_ptr, test_data))
    push!(results, test_dynamic_no_position_ptr_non_escaping(buf, position_ptr, test_data))
    
    # Advanced allocation tests (Tests 14-20)
    push!(results, test_multiple_buffer_reuse_escaping(buf, position_ptr, test_data))
    push!(results, test_large_vs_small_array_escaping(buf, position_ptr))
    push!(results, test_large_array_500x500(buf, position_ptr, test_data))  # Test 15b
    push!(results, test_type_unstable_return(buf, position_ptr, test_data))
    push!(results, test_nested_function_calls(buf, position_ptr, test_data))
    push!(results, test_global_vs_anonymous_function_calls(buf, position_ptr, test_data))  # Test 17b
    push!(results, test_exception_handling(buf, position_ptr, test_data))
    push!(results, test_multiple_array_operations(buf, position_ptr, test_data))
    push!(results, test_conditional_processing(buf, position_ptr, test_data))
    
    # Type instability deep dive tests (Tests 21-25)
    type_instability_results = run_type_instability_investigation(buf, position_ptr, test_data)
    for result in type_instability_results
        push!(results, result)
    end
    
    # Extract specific values for analysis
    allocs_escaping = results[1][2]  # First test (escaping)
    allocs_non_escaping = results[2][2]  # Second test (non-escaping)
    
    # Generate enhanced summary and analysis
    return generate_enhanced_summary_and_analysis(results, allocs_escaping, allocs_non_escaping)
end

# Test 21: Type stability comparison - Dynamic vs Explicit
function test_type_stability_comparison(buf, position_ptr, test_data)
    println("🧪 Test 21: Type Stability Comparison - Dynamic vs Explicit")
    
    # Pre-encode data into buffer (outside measured function)
    position_ptr[] = 0
    enc = TensorMessageEncoder(buf; position_ptr)
    encode(enc, test_data)
    
    # Test A: Dynamic decode (type-unstable)
    function decode_dynamic_unstable()
        position_ptr[] = 0
        dec = TensorMessageDecoder(buf; position_ptr)
        matrix = decode(dec)  # Dynamic - type unstable
        
        total = 0.0f0
        for val in matrix
            total += val
        end
        return total
    end
    
    # Test B: Explicit type decode (type-stable)
    function decode_explicit_stable()
        position_ptr[] = 0
        dec = TensorMessageDecoder(buf; position_ptr)
        matrix = decode(dec, Matrix{Float32})  # Explicit type - stable
        
        total = 0.0f0
        for val in matrix
            total += val
        end
        return total
    end
    
    # Warm up both
    result_dynamic = decode_dynamic_unstable()
    result_explicit = decode_explicit_stable()
    
    # Measure allocations
    allocs_dynamic = @allocated decode_dynamic_unstable()
    allocs_explicit = @allocated decode_explicit_stable()
    
    println("   Dynamic decode allocations:  $allocs_dynamic bytes")
    println("   Explicit decode allocations: $allocs_explicit bytes")
    println("   Type stability savings:      $(allocs_dynamic - allocs_explicit) bytes")
    println("   Results match: $(result_dynamic ≈ result_explicit)")
    println("   Reason: Direct comparison of type stability impact")
    println()
    
    return ("type_stability_comparison", allocs_dynamic - allocs_explicit, "dynamic vs explicit type decode")
end

# Test 22: Type inference failure investigation
function test_type_inference_failure(buf, position_ptr, test_data)
    println("🧪 Test 22: Type Inference Failure Investigation")
    
    # Pre-encode data into buffer (outside measured function)
    position_ptr[] = 0
    enc = TensorMessageEncoder(buf; position_ptr)
    encode(enc, test_data)
    
    function decode_with_type_assertion()
        position_ptr[] = 0
        dec = TensorMessageDecoder(buf; position_ptr)
        matrix = decode(dec)  # Dynamic decode
        # Convert to Matrix{Float32} instead of type assertion to avoid type errors
        matrix_converted = Matrix{Float32}(matrix)
        
        total = 0.0f0
        for val in matrix_converted
            total += val
        end
        return total
    end
    
    function decode_with_collect()
        position_ptr[] = 0
        dec = TensorMessageDecoder(buf; position_ptr)
        matrix = decode(dec)  # Dynamic decode
        matrix_collected = collect(matrix)  # Force materialization
        
        total = 0.0f0
        for val in matrix_collected
            total += val
        end
        return total
    end
    
    # Warm up
    result_assertion = decode_with_type_assertion()
    result_collect = decode_with_collect()
    
    # Measure allocations
    allocs_assertion = @allocated decode_with_type_assertion()
    allocs_collect = @allocated decode_with_collect()
    
    println("   Type conversion allocations: $allocs_assertion bytes")
    println("   Collect() allocations: $allocs_collect bytes")
    println("   Results (sum): $result_assertion, $result_collect")
    println("   Reason: Different ways to handle dynamic decode type uncertainty")
    println()
    
    return ("Type Inference Failure", allocs_assertion, "Type conversion after dynamic decode")
end

# Test 23: Multiple dynamic calls - compounding effect
function test_multiple_dynamic_calls(buf, position_ptr, test_data)
    println("🧪 Test 23: Multiple Dynamic Calls - Compounding Effect")
    
    # Pre-encode data into buffer (outside measured function)
    position_ptr[] = 0
    enc = TensorMessageEncoder(buf; position_ptr)
    encode(enc, test_data)
    
    function decode_multiple_dynamic()
        results = Float32[]
        
        for i in 1:3  # Multiple dynamic decodes
            position_ptr[] = 0
            dec = TensorMessageDecoder(buf; position_ptr)
            matrix = decode(dec)  # Dynamic - each call unstable
            
            total = 0.0f0
            for val in matrix
                total += val
            end
            push!(results, total)
        end
        
        return sum(results)
    end
    
    # Warm up
    result_multiple = decode_multiple_dynamic()
    
    # Measure allocations
    allocs_multiple = @allocated decode_multiple_dynamic()
    println("   Multiple dynamic allocations: $allocs_multiple bytes")
    println("   Per-call average: $(round(allocs_multiple / 3, digits=1)) bytes")
    println("   Result (sum): $result_multiple")
    println("   Reason: Does type instability compound with multiple calls?")
    println()
    
    return ("multiple_dynamic", allocs_multiple, "multiple dynamic decode calls")
end

# Test 24: Return type annotation investigation
function test_return_type_annotation(buf, position_ptr, test_data)
    println("🧪 Test 24: Return Type Annotation Investigation")
    
    # Pre-encode data into buffer (outside measured function)
    position_ptr[] = 0
    enc = TensorMessageEncoder(buf; position_ptr)
    encode(enc, test_data)
    
    function decode_with_return_annotation()::Float32
        position_ptr[] = 0
        dec = TensorMessageDecoder(buf; position_ptr)
        matrix = decode(dec)  # Dynamic decode but with return type annotation
        
        total = 0.0f0
        for val in matrix
            total += val
        end
        return total
    end
    
    # Warm up
    result_annotation = decode_with_return_annotation()
    
    # Measure allocations
    allocs_annotation = @allocated decode_with_return_annotation()
    println("   Return annotation allocations: $allocs_annotation bytes")
    println("   Result (sum): $result_annotation")
    println("   Reason: Does return type annotation help with dynamic decode?")
    println()
    
    return ("return_annotation", allocs_annotation, "return type annotation with dynamic decode")
end

# Test 25: Code_warntype investigation helper
function test_code_warntype_patterns(buf, position_ptr, test_data)
    println("🧪 Test 25: Code Analysis - Type Instability Patterns")
    
    # Pre-encode data into buffer (outside measured function)
    position_ptr[] = 0
    enc = TensorMessageEncoder(buf; position_ptr)
    encode(enc, test_data)
    
    # Function for @code_warntype analysis - Dynamic
    function unstable_for_analysis()
        position_ptr[] = 0
        dec = TensorMessageDecoder(buf; position_ptr)
        matrix = decode(dec)  # This should show type instability
        
        total = 0.0f0
        for val in matrix
            total += val
        end
        return total
    end
    
    # Function for @code_warntype analysis - Stable
    function stable_for_analysis()
        position_ptr[] = 0
        dec = TensorMessageDecoder(buf; position_ptr)
        matrix = decode(dec, Matrix{Float32})  # This should be stable
        
        total = 0.0f0
        for val in matrix
            total += val
        end
        return total
    end
    
    # Warm up
    result_unstable = unstable_for_analysis()
    result_stable = stable_for_analysis()
    
    # Measure allocations
    allocs_unstable = @allocated unstable_for_analysis()
    allocs_stable = @allocated stable_for_analysis()
    
    println("   Unstable function allocations: $allocs_unstable bytes")
    println("   Stable function allocations:   $allocs_stable bytes")
    println("   Type stability difference:     $(allocs_unstable - allocs_stable) bytes")
    println("   Results match: $(result_unstable ≈ result_stable)")
    
    # Provide code analysis hints
    println()
    println("   💡 To analyze type instability:")
    println("      @code_warntype unstable_for_analysis()")
    println("      @code_warntype stable_for_analysis()")
    println("   Reason: Compare compiler analysis between stable and unstable versions")
    println()
    
    return ("code_analysis", allocs_unstable - allocs_stable, "type instability investigation")
end

# Test 21: Direct comparison - Stable vs Unstable type patterns
function test_type_stability_comparison(buf, position_ptr, test_data)
    println("🧪 Test 21: Type Stability Direct Comparison")
    
    # Stable version - explicit type
    function stable_decode_explicit_type(buffer, ptr_ref)
        ptr_ref[] = 0
        dec = TensorMessageDecoder(buffer; position_ptr=ptr_ref)
        result = decode(dec, Matrix{Float32})  # Explicit type - STABLE
        return sum(result)
    end
    
    # Unstable version - inferred type
    function unstable_decode_inferred_type(buffer, ptr_ref)
        ptr_ref[] = 0
        dec = TensorMessageDecoder(buffer; position_ptr=ptr_ref)
        result = decode(dec)  # No type specified - UNSTABLE
        return sum(result)
    end
    
    # Warm up
    stable_result = stable_decode_explicit_type(buf, position_ptr)
    unstable_result = unstable_decode_inferred_type(buf, position_ptr)
    
    # Measure allocations
    allocs_stable = @allocated stable_decode_explicit_type(buf, position_ptr)
    allocs_unstable = @allocated unstable_decode_inferred_type(buf, position_ptr)
    
    println("   Stable decode (explicit type):   $allocs_stable bytes")
    println("   Unstable decode (inferred type): $allocs_unstable bytes")
    println("   Type instability penalty:        $(allocs_unstable - allocs_stable) bytes")
    println("   Results match: $(stable_result ≈ unstable_result)")
    println("   Reason: decode() without explicit type causes runtime type dispatch")
    println()
    
    return ("type_stability_comparison", allocs_unstable - allocs_stable, "stable vs unstable decode patterns")
end

# Test 22: Type inference failure with dynamic calls
function test_type_inference_failure(buf, position_ptr, test_data)
    println("🧪 Test 22: Type Inference Failure Analysis")
    
    # Function that defeats Julia's type inference
    function dynamic_type_selection(use_float32::Bool)
        if use_float32
            return Matrix{Float32}
        else
            return Matrix{Float64}
        end
    end
    
    function decode_with_dynamic_type(buffer, ptr_ref, use_f32)
        ptr_ref[] = 0
        dec = TensorMessageDecoder(buffer; position_ptr=ptr_ref)
        target_type = dynamic_type_selection(use_f32)  # Compiler can't infer this!
        result = decode(dec, target_type)
        return sum(result)
    end
    
    # Baseline stable version
    function decode_with_static_type(buffer, ptr_ref)
        ptr_ref[] = 0
        dec = TensorMessageDecoder(buffer; position_ptr=ptr_ref)
        result = decode(dec, Matrix{Float32})  # Compiler knows this at compile time
        return sum(result)
    end
    
    # Warm up
    dynamic_result = decode_with_dynamic_type(buf, position_ptr, true)
    static_result = decode_with_static_type(buf, position_ptr)
    
    # Measure allocations
    allocs_dynamic = @allocated decode_with_dynamic_type(buf, position_ptr, true)
    allocs_static = @allocated decode_with_static_type(buf, position_ptr)
    
    println("   Dynamic type selection: $allocs_dynamic bytes")
    println("   Static type selection:  $allocs_static bytes")
    println("   Dynamic type penalty:   $(allocs_dynamic - allocs_static) bytes")
    println("   Results match: $(dynamic_result ≈ static_result)")
    println("   Reason: Runtime type dispatch prevents compiler optimizations")
    println()
    
    return ("type_inference_failure", allocs_dynamic - allocs_static, "dynamic vs static type selection")
end

# Test 23: Multiple dynamic calls compounding type instability
function test_multiple_dynamic_calls(buf, position_ptr, test_data)
    println("🧪 Test 23: Multiple Dynamic Calls - Compounding Instability")
    
    # Function with multiple unstable calls
    function multiple_unstable_operations(buffer, ptr_ref)
        ptr_ref[] = 0
        dec1 = TensorMessageDecoder(buffer; position_ptr=ptr_ref)
        result1 = decode(dec1)  # First unstable call
        
        # Re-encode (simulating real-world pipeline)
        ptr_ref[] = 0
        enc = TensorMessageEncoder(buffer; position_ptr=ptr_ref)
        encode(enc, result1)
        
        # Decode again with different pattern
        ptr_ref[] = 0
        dec2 = TensorMessageDecoder(buffer; position_ptr=ptr_ref)
        result2 = decode(dec2)  # Second unstable call
        
        return sum(result2)
    end
    
    # Stable equivalent
    function multiple_stable_operations(buffer, ptr_ref)
        ptr_ref[] = 0
        dec1 = TensorMessageDecoder(buffer; position_ptr=ptr_ref)
        result1 = decode(dec1, Matrix{Float32})  # Explicit type
        
        # Re-encode
        ptr_ref[] = 0
        enc = TensorMessageEncoder(buffer; position_ptr=ptr_ref)
        encode(enc, result1)
        
        # Decode again with explicit type
        ptr_ref[] = 0
        dec2 = TensorMessageDecoder(buffer; position_ptr=ptr_ref)
        result2 = decode(dec2, Matrix{Float32})  # Explicit type
        
        return sum(result2)
    end
    
    # Warm up
    unstable_result = multiple_unstable_operations(buf, position_ptr)
    stable_result = multiple_stable_operations(buf, position_ptr)
    
    # Measure allocations
    allocs_unstable = @allocated multiple_unstable_operations(buf, position_ptr)
    allocs_stable = @allocated multiple_stable_operations(buf, position_ptr)
    
    println("   Multiple unstable calls: $allocs_unstable bytes")
    println("   Multiple stable calls:   $allocs_stable bytes")
    println("   Compounding instability: $(allocs_unstable - allocs_stable) bytes")
    println("   Results match: $(unstable_result ≈ stable_result)")
    println("   Reason: Each unstable call compounds allocation overhead")
    println()
    
    return ("multiple_dynamic_calls", allocs_unstable - allocs_stable, "compounding type instability effects")
end

# Test 24: Return type annotation effects
function test_return_type_annotation(buf, position_ptr, test_data)
    println("🧪 Test 24: Return Type Annotation Effects")
    
    # Without return type annotation
    function decode_no_annotation(buffer, ptr_ref)
        ptr_ref[] = 0
        dec = TensorMessageDecoder(buffer; position_ptr=ptr_ref)
        result = decode(dec)  # Compiler doesn't know return type
        return sum(result)  # Return type unknown until runtime
    end
    
    # With explicit return type annotation
    function decode_with_annotation(buffer, ptr_ref)::Float32
        ptr_ref[] = 0
        dec = TensorMessageDecoder(buffer; position_ptr=ptr_ref)
        result = decode(dec)  # Still unstable, but return type is known
        return sum(result)  # Compiler knows this returns Float32
    end
    
    # Fully stable version for comparison
    function decode_fully_stable(buffer, ptr_ref)::Float32
        ptr_ref[] = 0
        dec = TensorMessageDecoder(buffer; position_ptr=ptr_ref)
        result = decode(dec, Matrix{Float32})  # Both decode and return are typed
        return sum(result)
    end
    
    # Warm up
    no_annotation_result = decode_no_annotation(buf, position_ptr)
    with_annotation_result = decode_with_annotation(buf, position_ptr)
    fully_stable_result = decode_fully_stable(buf, position_ptr)
    
    # Measure allocations
    allocs_no_annotation = @allocated decode_no_annotation(buf, position_ptr)
    allocs_with_annotation = @allocated decode_with_annotation(buf, position_ptr)
    allocs_fully_stable = @allocated decode_fully_stable(buf, position_ptr)
    
    println("   No type annotation:     $allocs_no_annotation bytes")
    println("   Return type annotation: $allocs_with_annotation bytes")
    println("   Fully stable:           $allocs_fully_stable bytes")
    println("   Annotation benefit:     $(allocs_no_annotation - allocs_with_annotation) bytes")
    println("   Full stability benefit: $(allocs_no_annotation - allocs_fully_stable) bytes")
    println("   All results match: $(no_annotation_result ≈ with_annotation_result ≈ fully_stable_result)")
    println("   Reason: Return type annotations help but don't solve underlying type instability")
    println()
    
    return ("return_type_annotation", allocs_no_annotation - allocs_fully_stable, "type annotation optimization effects")
end

# Test 25: @code_warntype analysis patterns
function test_code_warntype_patterns(buf, position_ptr, test_data)
    println("🧪 Test 25: @code_warntype Analysis Patterns")
    
    # Function for @code_warntype analysis - Unstable
    function unstable_for_analysis()
        position_ptr[] = 0
        dec = TensorMessageDecoder(buf; position_ptr)
        matrix = decode(dec)  # This will show red warnings
        
        total = 0.0
        for val in matrix  # This will be type-unstable
            total += val
        end
        return total
    end
    
    # Function for @code_warntype analysis - Stable
    function stable_for_analysis()
        position_ptr[] = 0
        dec = TensorMessageDecoder(buf; position_ptr)
        matrix = decode(dec, Matrix{Float32})  # This should be stable
        
        total = 0.0f0
        for val in matrix
            total += val
        end
        return total
    end
    
    # Warm up
    result_unstable = unstable_for_analysis()
    result_stable = stable_for_analysis()
    
    # Measure allocations
    allocs_unstable = @allocated unstable_for_analysis()
    allocs_stable = @allocated stable_for_analysis()
    
    println("   Unstable function allocations: $allocs_unstable bytes")
    println("   Stable function allocations:   $allocs_stable bytes")
    println("   Type stability difference:     $(allocs_unstable - allocs_stable) bytes")
    println("   Results match: $(result_unstable ≈ result_stable)")
    
    # Provide code analysis hints
    println()
    println("   💡 To analyze type instability:")
    println("      @code_warntype unstable_for_analysis()")
    println("      @code_warntype stable_for_analysis()")
    println("   Reason: Compare compiler analysis between stable and unstable versions")
    println()
    
    return ("code_analysis", allocs_unstable - allocs_stable, "type instability investigation")
end

function run_type_instability_investigation(buf, position_ptr, test_data)
    # Run detailed type instability investigation tests (Tests 21-25)
    results = []
    push!(results, test_type_stability_comparison(buf, position_ptr, test_data))
    push!(results, test_type_inference_failure(buf, position_ptr, test_data))
    push!(results, test_multiple_dynamic_calls(buf, position_ptr, test_data))
    push!(results, test_return_type_annotation(buf, position_ptr, test_data))
    push!(results, test_code_warntype_patterns(buf, position_ptr, test_data))
    return results
end

function test_type_instability_deep_dive()
    # Run focused investigation into type instability allocations
    println("🔍 Type Instability Deep Dive Investigation")
    println("=" ^ 45)
    
    buf, position_ptr, test_data = setup_test_data()
    println("✓ Test data encoded: $(size(test_data)) matrix")
    println()
    
    results = run_type_instability_investigation(buf, position_ptr, test_data)
    
    # Find the worst type instability pattern
    max_allocs = results[1]
    for result in results
        if result[2] > max_allocs[2]
            max_allocs = result
        end
    end
    
    println()
    println("🚨 Type Instability Analysis:")
    println("   Worst pattern: $(max_allocs[1]) - $(max_allocs[2]) bytes")
    println("   Critical finding: Dynamic decode() causes massive allocations")
    println("   Solution: Always use explicit types: decode(dec, Matrix{Float32})")
    
    return generate_summary_and_analysis(results, max_allocs[2], 0)
end

if abspath(PROGRAM_FILE) == @__FILE__
    # Run the full test suite by default
    test_escaping_vs_non_escaping()
    
    # Uncomment any of these to run specific test categories:
    # test_minimal_core_only()
    # test_core_escape_analysis_only()
    # test_closure_capture_only()
    # test_buffer_allocation_only()
    # test_advanced_allocation_patterns()
    # test_optimization_comparison()
    # test_closure_impact_analysis()
    # test_type_instability_deep_dive()  # Focused investigation into type instability (Tests 21-25)
end
