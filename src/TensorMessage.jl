include("../ext/tensor/Tensor.jl")

import .Tensor: TensorMessage, TensorMessageDecoder, TensorMessageEncoder,
    values, values!, shape, shape!, offset, offset!, format

export MessageHeader, TensorMessage, TensorMessageDecoder, TensorMessageEncoder,
    values, values!, values!, shape, shape!, offset, offset!, format

using LinearAlgebra

function Base.eltype(T::Tensor.Format.SbeEnum)
    T == Tensor.Format.UINT8 ? UInt8 :
    T == Tensor.Format.INT8 ? Int8 :
    T == Tensor.Format.UINT16 ? UInt16 :
    T == Tensor.Format.INT16 ? Int16 :
    T == Tensor.Format.UINT32 ? UInt32 :
    T == Tensor.Format.INT32 ? Int32 :
    T == Tensor.Format.UINT64 ? UInt64 :
    T == Tensor.Format.INT64 ? Int64 :
    T == Tensor.Format.FLOAT32 ? Float32 :
    T == Tensor.Format.FLOAT64 ? Float64 :
    T == Tensor.Format.BOOLEAN ? Bool :
    throw(ArgumentError("unexpected format"))
end

function inv_eltype(T::Type{<:Real})
    T == UInt8 ? Tensor.Format.UINT8 :
    T == Int8 ? Tensor.Format.INT8 :
    T == UInt16 ? Tensor.Format.UINT16 :
    T == Int16 ? Tensor.Format.INT16 :
    T == UInt32 ? Tensor.Format.UINT32 :
    T == Int32 ? Tensor.Format.INT32 :
    T == UInt64 ? Tensor.Format.UINT64 :
    T == Int64 ? Tensor.Format.INT64 :
    T == Float32 ? Tensor.Format.FLOAT32 :
    T == Float64 ? Tensor.Format.FLOAT64 :
    T == Bool ? Tensor.Format.BOOLEAN :
    throw(ArgumentError("unexpected type"))
end

order(::Type{<:Adjoint}) = Tensor.Order.ROW
order(::Type{<:Transpose}) = Tensor.Order.ROW
order(::Type{<:AbstractArray}) = Tensor.Order.COLUMN

abstract type AbstractTensorMessageArray{T,N,O} end

# Type-stable
@inline function values(::Type{<:AbstractTensorMessageArray{T,N,O}}, m::Tensor.TensorMessageDecoder) where {T,N,O}
    Tensor.sbe_rewind!(m)

    type = Base.eltype(Tensor.format(m))
    T == type || error("Unexpected data type, expected $T but got $type")

    order = Tensor.order(m)
    O == order || error("Unexpected order, expected $O but got $order")

    shape = Tensor.shape(NTuple{N}, m)
    Tensor.skip_offset!(m)

    data = Tensor.values(T, m)
    array = reshape(data, shape)

    return O == Tensor.Order.COLUMN ? array : array'
end

# Type-stable
@inline function values!(::Type{<:AbstractTensorMessageArray{T,N,O}},
    m::Tensor.TensorMessageEncoder, shape::NTuple{N,Int}, offset::Union{NTuple{N,Int}, Nothing}=nothing) where {T,N,O}
    Tensor.sbe_rewind!(m)
    Tensor.format!(m, inv_eltype(T))
    Tensor.order!(m, O)
    Tensor.shape!(m, shape)
    if offset !== nothing
        Tensor.offset!(m, offset)
    else
        Tensor.skip_offset!(m)
    end
    data = Tensor.values!(T, m, prod(shape))
    array = reshape(data, shape)

    return O == Tensor.Order.COLUMN ? array : array'
end

@inline function values!(t::Type{<:AbstractTensorMessageArray{T,N,O}},
    m::Tensor.TensorMessageEncoder, src::AbstractArray{T,N}, offset=nothing) where {T,N,O}
    values!(t, m, size(src), offset) .= src
end

function Tensor.values!(m::Tensor.TensorMessageEncoder, src::A, offset=nothing) where {T,N,A<:AbstractArray{T,N}}
    values!(AbstractTensorMessageArray{T,N,order(A)}, m, src, offset)
end

@inline Tensor.shape(T::Type{<:Real}, m::Tensor.TensorMessageDecoder) = reinterpret(T, Tensor.shape(m))
@inline Tensor.shape(::Type{NTuple}, m::Tensor.TensorMessageDecoder) = Tuple(Tensor.shape(Int32, m))
function Tensor.shape(::Type{NTuple{N}}, m::Tensor.TensorMessageDecoder) where {N}
    shape = Tensor.shape(Int32, m)
    ntuple(i -> shape[i], Val(N))
end
@inline Tensor.shape!(T::Type{<:Real}, m::Tensor.TensorMessageEncoder, len::Int) = reinterpret(T, Tensor.shape!(m, sizeof(T) * len))
@inline Tensor.shape!(m::Tensor.TensorMessageEncoder, shape::NTuple{N,Int}) where {N} = Tensor.shape!(Int32, m, length(shape)) .= shape

@inline Tensor.offset(T::Type{<:Real}, m::Tensor.TensorMessageDecoder) = reinterpret(T, Tensor.offset(m))
@inline Tensor.offset(::Type{NTuple}, m::Tensor.TensorMessageDecoder) = Tuple(Tensor.offset(Int32, m))
@inline function Tensor.offset(::Type{NTuple{N}}, m::Tensor.TensorMessageDecoder) where {N}
    offset = Tensor.offset(Int32, m)
    ntuple(i -> offset[i], Val(N))
end
@inline Tensor.offset!(T::Type{<:Real}, m::Tensor.TensorMessageEncoder, len::Int) = reinterpret(T, Tensor.offset!(m, sizeof(T) * len))
@inline Tensor.offset!(m::Tensor.TensorMessageEncoder, offset::NTuple{N,Int}) where {N} = Tensor.offset!(Int32, m, length(offset)) .= offset

"""
    Tensor.values(::Type{<:AbstractArray}, m::Tensor.TensorMessageDecoder)

Reads the values from a TensorMessage and returns an array defined in the serialized buffer.
Note: This is not type-stable
"""
@inline function Tensor.values(::Type{<:AbstractArray}, m::Tensor.TensorMessageDecoder)
    Tensor.sbe_rewind!(m)
    T = Base.eltype(Tensor.format(m))
    O = Tensor.order(m)
    N = div(Int(Tensor.shape_length(m)), sizeof(Int32))

    return values(AbstractTensorMessageArray{T,N,O}, m)
end

"""
    Tensor.values(::Type{<:AbstractArray}, m::Tensor.TensorMessageDecoder)

Reads the values from a TensorMessage and returns an array defined in the serialized buffer.

Example:
```julia
message = Tensor.TensorMessageDecoder(Tensor.SbeCodecContext(), buf, Tensor.MessageHeader(buf))
values = Tensor.values(Matrix{UInt16}, message)
```
"""
@inline function Tensor.values(::Type{A}, m::Tensor.TensorMessageDecoder) where {T,N,A<:AbstractArray{T,N}}
    values(AbstractTensorMessageArray{T,N,order(A)}, m)
end

"""
    Tensor.values!(::Type{A}, m::Tensor.TensorMessageEncoder, shape::NTuple{N}, offset::NTuple{N}=(0, 0)) where {T,N,A<:AbstractArray{T,N}}

Returns a type-stable array defined in the serialized buffer.

Example:
```julia
message = Tensor.TensorMessageEncoder(Tensor.SbeCodecContext(), buf, Tensor.MessageHeader(buf))
values = Tensor.values!(Matrix{UInt16}, message, (2, 3))
```
"""
@inline function Tensor.values!(::Type{A}, m::Tensor.TensorMessageEncoder, shape, offset=nothing) where {T,N,A<:AbstractArray{T,N}}
    values!(AbstractTensorMessageArray{T,N,order(A)}, m, shape, offset)
end

@inline Tensor.values(T::Type{<:Real}, m::Tensor.TensorMessageDecoder) = reinterpret(T, Tensor.values(m))
@inline Tensor.values!(T::Type{<:Real}, m::Tensor.TensorMessageEncoder, len) = reinterpret(T, Tensor.values!(m, sizeof(T) * len))

function Base.show(io::IO, m::Tensor.TensorMessage{T}) where {T}
    println(io, "TensorMessage view over a type $T")
    println(io, "SbeBlockLength: ", Tensor.sbe_block_length(m))
    println(io, "SbeTemplateId:  ", Tensor.sbe_template_id(m))
    println(io, "SbeSchemaId:    ", Tensor.sbe_schema_id(m))
    println(io, "SbeSchemaVersion: ", Tensor.sbe_schema_version(m))

    writer = Tensor.TensorMessageDecoder(Tensor.SbeCodecContext(), Tensor.sbe_buffer(m), Tensor.sbe_offset(m),
        Tensor.sbe_block_length(m), Tensor.sbe_schema_version(m))
    print(io, "header: ")
    show(io, Tensor.header(writer))

    println(io)
    print(io, "format: ")
    print(io, Tensor.format(writer))

    println(io)
    print(io, "order: ")
    print(io, Tensor.order(writer))

    println(io)
    print(io, "shape: ")
    print(io, Tensor.shape(NTuple, writer))

    println(io)
    print(io, "offset: ")
    print(io, Tensor.offset(NTuple, writer))

    println(io)
    print(io, "values: ")
    print(io, Tensor.skip_values!(writer))
    print(io, " bytes of raw data")

    nothing
end