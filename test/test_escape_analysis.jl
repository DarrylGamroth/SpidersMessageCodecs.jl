#!/usr/bin/env julia

"""
Test to demonstrate allocation behavior with escaping vs non-escaping decoded values
"""

using SpidersMessageCodecs
using LinearAlgebra

function test_escaping_vs_non_escaping()
    println("🔍 Testing Allocation Behavior: Escaping vs Non-Escaping")
    println("=" ^ 60)
    
    # Setup test data
    buf = zeros(UInt8, 2000)
    position_ptr = Ref{Int64}(0)
    test_data = Float32[1.0 2.0 3.0; 4.0 5.0 6.0]
    
    # Encode test data once
    enc = TensorMessageEncoder(buf; position_ptr)
    encode(enc, test_data)
    
    println("✓ Test data encoded: $(size(test_data)) matrix")
    println()
    
    # Test 1: Standard decode that ESCAPES (returns the array)
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
    
    # Test 2: Decode that DOESN'T ESCAPE (processes in-place)
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
    
    # Test 3: Decode into pre-allocated buffer that DOESN'T ESCAPE
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
    
    # Test 4: Direct processing without intermediate arrays
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
    
    # Test 5: Simple iteration pattern
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
    
    # Test 6: Dynamic decode - ESCAPING (returns array, type inferred)
    println("🧪 Test 6: Dynamic decode - ESCAPING (auto type inference)")
    
    function decode_dynamic_escaping()
        position_ptr[] = 0
        dec = TensorMessageDecoder(buf; position_ptr)
        return decode(dec)  # No type specified - dynamic inference
    end
    
    # Warm up
    result_dynamic_escaping = decode_dynamic_escaping()
    
    # Measure allocations
    allocs_dynamic_escaping = @allocated decode_dynamic_escaping()
    println("   Allocations: $allocs_dynamic_escaping bytes")
    println("   Result type: $(typeof(result_dynamic_escaping))")
    println("   Result size: $(size(result_dynamic_escaping))")
    println("   Reason: Dynamic type inference may add overhead")
    println()
    
    # Test 7: Dynamic decode - NON-ESCAPING (processes in-place)
    println("🧪 Test 7: Dynamic decode - NON-ESCAPING (processes in-place)")
    
    function decode_dynamic_non_escaping()
        position_ptr[] = 0
        dec = TensorMessageDecoder(buf; position_ptr)
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
    
    # Test 8: Dynamic decode with type assertion
    println("🧪 Test 8: Dynamic decode with type assertion")
    
    function decode_dynamic_assert()
        position_ptr[] = 0
        dec = TensorMessageDecoder(buf; position_ptr)
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
    println("   Reason: Dynamic decode with type assertion for optimization")
    println()

    # Test 9: Encoder/Decoder without position_ptr - ESCAPING
    println("🧪 Test 9: No position_ptr - ESCAPING (returns array)")
    
    function encode_decode_no_ptr_escaping()
        buf = zeros(UInt8, 2000)
        
        # Encode without position_ptr
        enc = TensorMessageEncoder(buf)
        encode(enc, test_data)
        
        # Decode without position_ptr
        dec = TensorMessageDecoder(buf)
        return decode(dec, Matrix{Float32})  # Array escapes function
    end
    
    # Warm up
    result_no_ptr_escaping = encode_decode_no_ptr_escaping()
    
    # Measure allocations
    allocs_no_ptr_escaping = @allocated encode_decode_no_ptr_escaping()
    println("   Allocations: $allocs_no_ptr_escaping bytes")
    println("   Result size: $(size(result_no_ptr_escaping))")
    println("   Reason: No position_ptr parameter - simpler API")
    println()
    
    # Test 10: Encoder/Decoder without position_ptr - NON-ESCAPING
    println("🧪 Test 10: No position_ptr - NON-ESCAPING (processes in-place)")
    
    function encode_decode_no_ptr_non_escaping()
        buf = zeros(UInt8, 2000)
        
        # Encode without position_ptr
        enc = TensorMessageEncoder(buf)
        encode(enc, test_data)
        
        # Decode without position_ptr
        dec = TensorMessageDecoder(buf)
        matrix = decode(dec, Matrix{Float32})
        
        # Process the array without returning it (doesn't escape)
        total = 0.0f0
        for val in matrix
            total += val
        end
        
        return total  # Only scalar escapes, not the array
    end
    
    # Warm up
    result_no_ptr_non_escaping = encode_decode_no_ptr_non_escaping()
    
    # Measure allocations
    allocs_no_ptr_non_escaping = @allocated encode_decode_no_ptr_non_escaping()
    println("   Allocations: $allocs_no_ptr_non_escaping bytes")
    println("   Result (sum): $result_no_ptr_non_escaping")
    println("   Expected: $(sum(test_data))")
    println("   Reason: No position_ptr - escape analysis with buffer allocation")
    println()
    
    # Test 11: Dynamic decode without position_ptr - ESCAPING
    println("🧪 Test 11: Dynamic no position_ptr - ESCAPING (auto type inference)")
    
    function encode_decode_dynamic_no_ptr_escaping()
        buf = zeros(UInt8, 2000)
        
        # Encode without position_ptr
        enc = TensorMessageEncoder(buf)
        encode(enc, test_data)
        
        # Decode without position_ptr - dynamic
        dec = TensorMessageDecoder(buf)
        return decode(dec)  # No type specified - dynamic inference
    end
    
    # Warm up
    result_dynamic_no_ptr_escaping = encode_decode_dynamic_no_ptr_escaping()
    
    # Measure allocations
    allocs_dynamic_no_ptr_escaping = @allocated encode_decode_dynamic_no_ptr_escaping()
    println("   Allocations: $allocs_dynamic_no_ptr_escaping bytes")
    println("   Result type: $(typeof(result_dynamic_no_ptr_escaping))")
    println("   Result size: $(size(result_dynamic_no_ptr_escaping))")
    println("   Reason: Dynamic decode without position_ptr - includes buffer allocation")
    println()
    
    # Test 12: Dynamic decode without position_ptr - NON-ESCAPING
    println("🧪 Test 12: Dynamic no position_ptr - NON-ESCAPING (processes in-place)")
    
    function encode_decode_dynamic_no_ptr_non_escaping()
        buf = zeros(UInt8, 2000)
        
        # Encode without position_ptr
        enc = TensorMessageEncoder(buf)
        encode(enc, test_data)
        
        # Decode without position_ptr - dynamic
        dec = TensorMessageDecoder(buf)
        matrix = decode(dec)  # Dynamic decode
        
        # Process the array without returning it (doesn't escape)
        total = 0.0
        for val in matrix
            total += val
        end
        
        return total  # Only scalar escapes, not the array
    end
    
    # Warm up
    result_dynamic_no_ptr_non_escaping = encode_decode_dynamic_no_ptr_non_escaping()
    
    # Measure allocations
    allocs_dynamic_no_ptr_non_escaping = @allocated encode_decode_dynamic_no_ptr_non_escaping()
    println("   Allocations: $allocs_dynamic_no_ptr_non_escaping bytes")
    println("   Result (sum): $result_dynamic_no_ptr_non_escaping")
    println("   Expected: $(sum(test_data))")
    println("   Reason: Dynamic decode without position_ptr - full allocation overhead")
    println()
    
    # Summary
    println("📊 Summary of Allocation Behavior")
    println("-" ^ 50)
    println("Escaping decode:      $allocs_escaping bytes (must allocate)")
    println("Non-escaping decode:  $allocs_non_escaping bytes (may optimize away)")
    println("Buffer copy:          $allocs_buffer bytes (original doesn't escape)")
    println("Direct processing:    $allocs_direct bytes (no arrays)")
    println("Simple iteration:     $allocs_simple bytes (common pattern)")
    println("Dynamic escaping:     $allocs_dynamic_escaping bytes (type inferred)")
    println("Dynamic non-escaping: $allocs_dynamic_non_escaping bytes (process in-place)")
    println("Dynamic with assert:  $allocs_dynamic_assert bytes (type asserted)")
    println("No ptr escaping:      $allocs_no_ptr_escaping bytes (simpler API)")
    println("No ptr non-escaping:  $allocs_no_ptr_non_escaping bytes (buffer allocated)")
    println("Dynamic no ptr escaping: $allocs_dynamic_no_ptr_escaping bytes (dynamic, buffer)")
    println("Dynamic no ptr non-escaping: $allocs_dynamic_no_ptr_non_escaping bytes (dynamic, full)")
    println()
    
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
    println("💡 Key Insights:")
    println("   TYPED DECODE: decode(dec, Matrix{Float32})")
    println("   - When decoded values don't escape the function scope,")
    println("     Julia's compiler can eliminate allocations through escape analysis.")
    println("   - Type is known at compile time, enabling better optimization")
    println()
    println("   DYNAMIC DECODE: decode(dec)")
    println("   - Type is inferred at runtime, may add overhead")
    println("   - Type assertions (::Matrix{Float32}) can help optimization")
    println("   - Escape analysis still applies for non-escaping patterns")
    println()
    println("   This is why some operations may show 0 allocations even when")
    println("   using the standard decode() function, regardless of typing.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    test_escaping_vs_non_escaping()
end
