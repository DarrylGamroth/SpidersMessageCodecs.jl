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

# Create buffer and encoder/decoder
buf = zeros(UInt8, 10000)
position_ptr = Ref{Int64}(0)
data = rand(Float32, 100, 50)

# Encoding
enc = TensorMessageEncoder(buf; position_ptr)
encode(enc, data)

# Decoding
dec = TensorMessageDecoder(buf; position_ptr)
decoded = decode(dec, Matrix{Float32})
```

## API Reference

### Core Types

#### TensorMessageEncoder
```julia
# Basic constructor (with MessageHeader)
enc = TensorMessageEncoder(buffer::AbstractArray; position_ptr::Ref{Int64}=Ref(0))

# Low-level constructor (without MessageHeader)
enc = TensorMessageEncoder(buffer, offset, position_ptr, hasSbeHeader=false)
```

#### TensorMessageDecoder
```julia
# Basic constructor (with MessageHeader)
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
# Position management
pos = sbe_position(encoder_or_decoder)
sbe_position!(encoder_or_decoder, new_position)
sbe_rewind!(encoder_or_decoder)  # Reset to start of message

# Convert to bytes for transmission
bytes = convert(AbstractArray{UInt8}, encoder)
```

## Testing and Development

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
- [Aeron.jl](https://github.com/package/Aeron.jl) - High-performance messaging (compatible transport)
