#!/usr/bin/env julia

# Benchmark runner for SpidersMessageCodecs.jl
# Usage: julia --project=. benchmark/run_benchmarks.jl

using Pkg
Pkg.activate(dirname(@__DIR__))

using BenchmarkTools
using SpidersMessageCodecs
using LinearAlgebra
using Printf

include("benchmarks.jl")

function print_results(results)
    println("\n" * "="^80)
    println("SpidersMessageCodecs.jl Benchmark Results")
    println("="^80)
    
    for (group_name, group) in results
        println("\n📊 $group_name")
        println("-" * "^" * string(length(group_name) + 3))
        
        for (bench_name, result) in group
            time_str = BenchmarkTools.prettytime(minimum(result).time)
            allocs = minimum(result).allocs
            memory = BenchmarkTools.prettymemory(minimum(result).memory)
            
            @printf "  %-25s %10s  %3d allocs  %8s\n" bench_name time_str allocs memory
        end
    end
    
    println("\n" * "="^80)
end

function run_quick_benchmarks()
    println("🚀 Running quick benchmark suite...")
    
    # Create a subset for quick runs
    quick_suite = BenchmarkGroup()
    quick_suite["TensorMessage"] = BenchmarkGroup()
    quick_suite["EventMessage"] = BenchmarkGroup()
    
    # Small benchmarks only
    quick_suite["TensorMessage"]["encode_small"] = SUITE["TensorMessage"]["encode_small"]
    quick_suite["TensorMessage"]["decode_small_typestable"] = SUITE["TensorMessage"]["decode_small_typestable"]
    quick_suite["EventMessage"]["encode_string"] = SUITE["EventMessage"]["encode_string"]
    quick_suite["EventMessage"]["encode_number"] = SUITE["EventMessage"]["encode_number"]
    
    # Run with reduced samples for speed
    tune!(quick_suite)
    results = run(quick_suite, samples=10, seconds=2)
    print_results(results)
    
    return results
end

function run_full_benchmarks()
    println("🏁 Running full benchmark suite...")
    
    # Tune the benchmarks
    println("Tuning benchmarks...")
    tune!(SUITE)
    
    # Run the full suite
    results = run(SUITE, verbose=true)
    print_results(results)
    
    return results
end

function main()
    if length(ARGS) > 0 && ARGS[1] == "--quick"
        results = run_quick_benchmarks()
    else
        results = run_full_benchmarks()
    end
    
    # Performance regression warnings
    println("\n⚠️  Performance Notes:")
    println("  • Encoding should be sub-microsecond for small tensors")
    println("  • Zero-allocation encoding should have ≤10 allocations after warmup")
    println("  • Type-stable decoding should be faster than dynamic decoding")
    println("  • Memory usage should scale linearly with data size")
    
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
