# SpidersMessageCodecs.jl

SpidersMessageCodecs.jl is a Julia package for binary serialization of multi-dimensional arrays and structured data using Simple Binary Encoding (SBE). It was developed for the SPIDERS (Subaru Pathfinder Instrument for Detecting Exoplanets & Retrieving Spectra) astronomical instrument.

## Features

- Zero-allocation decoding through type-stable operations
- Compact binary format based on Simple Binary Encoding (SBE)
- Support for multi-dimensional arrays with metadata (dimensions, origins, data formats)
- Compatible with network protocols and memory-mapped files
- Package extensions for high-level array operations

## Usage

```julia
using SpidersMessageCodecs

# Create buffer and data
buf = zeros(UInt8, 10000)
data = rand(Float32, 100, 50)

# Encoding (recommended API)
enc = TensorMessageEncoder(buf)
encode(enc, data)
encoded_size = sbe_encoded_length(enc)

# Decoding (recommended API)
dec = TensorMessageDecoder(buf)
decoded = decode(dec, Matrix{Float32})  # Type-stable for zero allocations
```

## API Reference

### Core Types

#### TensorMessageEncoder
```julia
# Recommended constructor (simple API)
enc = TensorMessageEncoder(buffer::AbstractArray)

# Advanced constructor (for manual position tracking)
enc = TensorMessageEncoder(buffer::AbstractArray; position_ptr::Ref{Int64}=Ref(0))

# Low-level constructor (without MessageHeader)
enc = TensorMessageEncoder(buffer, offset, position_ptr, hasSbeHeader=false)
```

#### TensorMessageDecoder
```julia
# Recommended constructor (simple API)
dec = TensorMessageDecoder(buffer::AbstractArray)

# Advanced constructor (for manual position tracking)
dec = TensorMessageDecoder(buffer::AbstractArray; position_ptr::Ref{Int64}=Ref(0))

# Low-level constructor (without MessageHeader)
dec = TensorMessageDecoder(buffer, offset, position_ptr, acting_block_length, acting_version)
```

### Encoding Operations
```julia
# Encode array data (provided by extension)
encode(enc, data::AbstractArray; origin=nothing)

# Encode with specific array type (provided by extension)
encode(enc, Matrix{Float64}, dims; origin=nothing)
encode(enc, Array{Float32, 3}, dims...; origin=nothing)

# Low-level field access
SpidersMessageCodecs.format!(enc, SpidersMessageCodecs.Format.SBE)
SpidersMessageCodecs.majorOrder!(enc, SpidersMessageCodecs.MajorOrder.COLUMN)
SpidersMessageCodecs.dims!(enc, dimension_data)
SpidersMessageCodecs.origin!(enc, origin_data)
SpidersMessageCodecs.values!(enc, array_data)
```

### Decoding Operations
```julia
# Type-stable decoding (zero allocations possible, provided by extension)
data = decode(dec, Matrix{Float64})
data = decode(dec, Array{Float32, 3})

# Dynamic decoding (some allocations, provided by extension)
data = decode(dec)

# Low-level field access
format_val = SpidersMessageCodecs.format(dec)  # Returns SpidersMessageCodecs.Format.SbeEnum
major_order = SpidersMessageCodecs.majorOrder(dec)  # Returns SpidersMessageCodecs.MajorOrder.SbeEnum
dims_data = SpidersMessageCodecs.dims(dec, NTuple{N,Int32})
origin_data = SpidersMessageCodecs.origin(dec, AbstractArray{Int32})
values_data = SpidersMessageCodecs.values(dec, AbstractArray{T})
```

### Buffer Management
```julia
# Get encoded message size (recommended)
size = sbe_encoded_length(encoder)

# Position management (advanced usage)
pos = sbe_position(encoder_or_decoder)
sbe_position!(encoder_or_decoder, new_position)
sbe_rewind!(encoder_or_decoder)  # Reset to start of message

# Convert to bytes for transmission
bytes = convert(AbstractArray{UInt8}, encoder)
```

## Performance Recommendations

### For Zero-Allocation Performance

**Decoding:**
```julia
# ✅ RECOMMENDED: Use explicit types for zero allocations
data = decode(dec, Matrix{Float32})

# ❌ AVOID: Dynamic decoding causes allocations
data = decode(dec)  # Can allocate 160+ bytes
```

**Encoding:**
```julia
# ✅ RECOMMENDED: Reuse buffers for zero allocations
buf = zeros(UInt8, 10000)  # Allocate once
enc = TensorMessageEncoder(buf)  # Reuse buffer
encode(enc, data)

# ❌ AVOID: Fresh buffers cause allocations
fresh_buf = zeros(UInt8, 10000)  # New allocation each time
enc = TensorMessageEncoder(fresh_buf)  # ~5,160 bytes overhead
```

**API Choice:**
- Use the simple API (`TensorMessageEncoder(buf)`) for most cases
- Only use `position_ptr` when you need manual position tracking
- Both achieve the same zero-allocation performance

## Testing and Development

### Key Performance Insights

Our comprehensive allocation analysis revealed:

1. **Type Stability is Critical**: Using explicit types (`decode(dec, Matrix{Float32})`) enables Julia's escape analysis to achieve zero allocations
2. **Buffer Reuse Matters**: Reusing the same buffer for encoding achieves zero allocations, while fresh buffers add ~5,160 bytes overhead
3. **API Choice is Flexible**: Both simple API (`TensorMessageEncoder(buf)`) and advanced API (`TensorMessageEncoder(buf; position_ptr)`) achieve identical performance
4. **Closure Capture Impact**: Global variable capture in closures adds 700+ bytes overhead vs local variables

### Running Tests
```julia
# Run complete test suite
using Pkg; Pkg.test()

# Run specific test categories
julia --project test/test_core.jl              # Core functionality
julia --project test/test_performance.jl       # Performance tests
julia --project test/test_generator_patterns_fixed.jl  # Generator patterns
julia --project test/test_escape_analysis.jl   # Allocation analysis
```

### Performance Testing
The package includes comprehensive performance and allocation tests:

```julia
# Zero-allocation validation
julia --project test/test_zero_allocation_comprehensive.jl

# Generator pattern comparisons
julia --project test/test_generator_patterns_fixed.jl

# Escape analysis investigation
julia --project test/test_escape_analysis.jl
```

## Contributing

Contributions are welcome! Please ensure:

1. All tests pass: `julia --project -e "using Pkg; Pkg.test()"`
2. Performance benchmarks show no regression
3. New features include appropriate tests
4. Documentation is updated for API changes

## Related Projects

- [SBE (Simple Binary Encoding)](https://real-logic.github.io/simple-binary-encoding/) - The binary format specification
- [Aeron.jl](https://github.com/DarrylGamroth/Aeron.jl) - High-performance messaging (compatible transport)
