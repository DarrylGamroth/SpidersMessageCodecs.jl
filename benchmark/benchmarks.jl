using BenchmarkTools
using SpidersMessageCodecs
using LinearAlgebra

# Define benchmark suite
const SUITE = BenchmarkGroup()

# TensorMessage benchmarks
SUITE["TensorMessage"] = BenchmarkGroup()

# Small tensor benchmarks
let data = Float32[1.0 2.0; 3.0 4.0], buf = zeros(UInt8, 1000)
    SUITE["TensorMessage"]["encode_small"] = @benchmarkable begin
        position_ptr = Ref{Int64}(0)
        enc = TensorMessageEncoder($buf; position_ptr)
        encode(enc, $data)
    end
    
    SUITE["TensorMessage"]["decode_small_typestable"] = @benchmarkable begin
        position_ptr = Ref{Int64}(0)
        enc = TensorMessageEncoder($buf; position_ptr)
        encode(enc, $data)
        position_ptr[] = 0
        dec = TensorMessageDecoder($buf; position_ptr)
        decode(dec, Matrix{Float32})
    end
end

# Medium tensor benchmarks - use deterministic data instead of rand()
let data = Float32[i*j*0.01 for i in 1:100, j in 1:100], buf = zeros(UInt8, 50_000)
    SUITE["TensorMessage"]["encode_medium"] = @benchmarkable begin
        position_ptr = Ref{Int64}(0)
        enc = TensorMessageEncoder($buf; position_ptr)
        encode(enc, $data)
    end
    
    SUITE["TensorMessage"]["decode_medium_typestable"] = @benchmarkable begin
        position_ptr = Ref{Int64}(0)
        enc = TensorMessageEncoder($buf; position_ptr)
        encode(enc, $data)
        position_ptr[] = 0
        dec = TensorMessageDecoder($buf; position_ptr)
        decode(dec, Matrix{Float32})
    end
end

# Large tensor benchmarks - use deterministic data that fits in Int16 range
let data = Int16[(i + j) % 32767 for i in 1:1000, j in 1:1000], buf = zeros(UInt8, 2_100_000)
    SUITE["TensorMessage"]["encode_large"] = @benchmarkable begin
        position_ptr = Ref{Int64}(0)
        enc = TensorMessageEncoder($buf; position_ptr)
        encode(enc, $data)
    end
    
    SUITE["TensorMessage"]["decode_large_typestable"] = @benchmarkable begin
        position_ptr = Ref{Int64}(0)
        enc = TensorMessageEncoder($buf; position_ptr)
        encode(enc, $data)
        position_ptr[] = 0
        dec = TensorMessageDecoder($buf; position_ptr)
        decode(dec, Matrix{Int16})
    end
end

# Dynamic decoding benchmarks
let data = Float64[1.0 2.0 3.0; 4.0 5.0 6.0], buf = zeros(UInt8, 1000)
    SUITE["TensorMessage"]["decode_dynamic"] = @benchmarkable begin
        position_ptr = Ref{Int64}(0)
        enc = TensorMessageEncoder($buf; position_ptr)
        encode(enc, $data)
        position_ptr[] = 0
        decode($buf; position_ptr)
    end
end

# EventMessage benchmarks
SUITE["EventMessage"] = BenchmarkGroup()

let buf = zeros(UInt8, 1000)
    SUITE["EventMessage"]["encode_string"] = @benchmarkable begin
        position_ptr = Ref{Int64}(0)
        enc = EventMessageEncoder($buf; position_ptr)
        encode(enc, "Hello, World!")
    end
    
    SUITE["EventMessage"]["encode_number"] = @benchmarkable begin
        position_ptr = Ref{Int64}(0)
        enc = EventMessageEncoder($buf; position_ptr)
        encode(enc, 42.0)
    end
    
    SUITE["EventMessage"]["encode_boolean"] = @benchmarkable begin
        position_ptr = Ref{Int64}(0)
        enc = EventMessageEncoder($buf; position_ptr)
        encode(enc, true)
    end
end

# Memory allocation benchmarks - focusing on zero-allocation encoding
SUITE["Memory"] = BenchmarkGroup()

let data = Int32[1 2; 3 4], buf = zeros(UInt8, 1000)
    SUITE["Memory"]["zero_alloc_encode"] = @benchmarkable begin
        position_ptr = Ref{Int64}(0)
        enc = TensorMessageEncoder($buf; position_ptr)
        encode(enc, $data)
    end
    
    SUITE["Memory"]["minimal_alloc_decode"] = @benchmarkable begin
        position_ptr = Ref{Int64}(0)
        enc = TensorMessageEncoder($buf; position_ptr)
        encode(enc, $data)
        position_ptr[] = 0
        dec = TensorMessageDecoder($buf; position_ptr)
        decode(dec, Matrix{Int32})
    end
end

# Type stability benchmarks - use deterministic data instead of rand()
SUITE["TypeStability"] = BenchmarkGroup()

let data = Float32[i*j*0.1 for i in 1:10, j in 1:10], buf = zeros(UInt8, 5000)
    SUITE["TypeStability"]["encode_inferred"] = @benchmarkable begin
        position_ptr = Ref{Int64}(0)
        enc = TensorMessageEncoder($buf; position_ptr)
        encode(enc, $data)
    end
    
    SUITE["TypeStability"]["decode_inferred"] = @benchmarkable begin
        position_ptr = Ref{Int64}(0)
        enc = TensorMessageEncoder($buf; position_ptr)
        encode(enc, $data)
        position_ptr[] = 0
        dec = TensorMessageDecoder($buf; position_ptr)
        decode(dec, Matrix{Float32})  # Type-stable decode
    end
end
