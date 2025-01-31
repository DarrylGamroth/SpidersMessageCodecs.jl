function Sbe.message_type(
    ::Val{Tensor.sbe_template_id(Tensor.TensorMessage)},
    ::Val{Tensor.sbe_schema_id(Tensor.TensorMessage)})
    Tensor.TensorMessage
end

function Sbe.decoder(::Type{<:Tensor.TensorMessage}, buffer::AbstractArray, offset::Int64, position_ptr::Base.RefValue)
    Tensor.TensorMessageDecoder(buffer, offset, position_ptr, Tensor.MessageHeader(buffer, offset))
end

function Sbe.sbe_decoded_length(m::Tensor.TensorMessage)
    Tensor.sbe_decoded_length(m)
end

Sbe.is_sbe_message(::Type{<:Tensor.TensorMessage}) = true

function Base.convert(::Type{<:AbstractArray{UInt8}}, m::Tensor.TensorMessage)
    offset = Tensor.sbe_offset(m) - Tensor.sbe_encoded_length(Tensor.MessageHeader)
    offset < 0 && throw(ArgumentError("Message offset is negative"))
    len = Tensor.sbe_decoded_length(m)
    return view(Tensor.sbe_buffer(m), offset+1:Tensor.sbe_offset(m)+len)
end

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

function Base.convert(::Type{Tensor.Format.SbeEnum}, T::Type{<:Real})
    T === UInt8 ? Tensor.Format.UINT8 :
    T === Int8 ? Tensor.Format.INT8 :
    T === UInt16 ? Tensor.Format.UINT16 :
    T === Int16 ? Tensor.Format.INT16 :
    T === UInt32 ? Tensor.Format.UINT32 :
    T === Int32 ? Tensor.Format.INT32 :
    T === UInt64 ? Tensor.Format.UINT64 :
    T === Int64 ? Tensor.Format.INT64 :
    T === Float32 ? Tensor.Format.FLOAT32 :
    T === Float64 ? Tensor.Format.FLOAT64 :
    T === Bool ? Tensor.Format.BOOLEAN :
    throw(ArgumentError("unexpected type"))
end

order(::Type{<:Adjoint}) = Tensor.Order.ROW
order(::Type{<:Transpose}) = Tensor.Order.ROW
order(::Type{<:AbstractArray}) = Tensor.Order.COLUMN

abstract type AbstractTensorMessageArray{T,N,O} end

# Type-stable
@inline function values(
    ::Type{<:AbstractTensorMessageArray{T,N,O}},
    m::Tensor.TensorMessageDecoder) where {T,N,O}

    Tensor.sbe_rewind!(m)

    # FIXME these checks slow down the code
    # type = Base.eltype(Tensor.format(m))
    # T <: type || error("Unexpected data type, expected $T but got $type")

    # order = Tensor.order(m)
    # O == order || error("Unexpected order, expected $O but got $order")

    dims = Tensor.dims(NTuple{N}, m)
    Tensor.skip_offset!(m)

    data = Tensor.values(T, m)
    reshape(data, dims)
end

# Type-stable
@inline function values!(
    ::Type{<:AbstractTensorMessageArray{T,N,O}},
    m::Tensor.TensorMessageEncoder,
    dims::NTuple{N,Int};
    offset::Union{Nothing,NTuple{N,Int}}=nothing) where {T,N,O}

    Tensor.sbe_rewind!(m)

    Tensor.format!(m, convert(Tensor.Format.SbeEnum, T))
    Tensor.order!(m, O)

    Tensor.dims!(m, dims)
    if offset === nothing
        Tensor.offset!(m, 0)
    else
        Tensor.offset!(m, offset)
    end
    data = Tensor.values!(T, m, prod(dims))
    reshape(data, dims)
end

@inline function values!(
    t::Type{<:AbstractTensorMessageArray{T,N,O}},
    m::Tensor.TensorMessageEncoder,
    src::AbstractArray{T,N}; offset=nothing) where {T,N,O}
    values!(t, m, size(src); offset) .= src
end

@inline Tensor.dims(::Type{T}, m::Tensor.TensorMessageDecoder) where {T<:Real} = reinterpret(T, Tensor.dims(m))
@inline Tensor.dims(::Type{NTuple}, m::Tensor.TensorMessageDecoder) = Tuple(Tensor.dims(Int32, m))
@inline function Tensor.dims(::Type{NTuple{N}}, m::Tensor.TensorMessageDecoder) where {N}
    dims = Tensor.dims(Int32, m)
    ntuple(i -> dims[i], Val(N))
end
@inline Tensor.dims!(::Type{T}, m::Tensor.TensorMessageEncoder, len::Int) where {T<:Real} = reinterpret(T, Tensor.dims!(m, sizeof(T) * len))
@inline Tensor.dims!(m::Tensor.TensorMessageEncoder, dims::NTuple{N,Int}) where {N} = Tensor.dims!(Int32, m, length(dims)) .= dims

@inline Tensor.offset(::Type{T}, m::Tensor.TensorMessageDecoder) where {T<:Real} = reinterpret(T, Tensor.offset(m))
@inline Tensor.offset(::Type{NTuple}, m::Tensor.TensorMessageDecoder) = Tuple(Tensor.offset(Int32, m))
@inline function Tensor.offset(::Type{NTuple{N}}, m::Tensor.TensorMessageDecoder) where {N}
    offset = Tensor.offset(Int32, m)
    ntuple(i -> offset[i], Val(N))
end
@inline Tensor.offset!(::Type{T}, m::Tensor.TensorMessageEncoder, len::Int) where {T<:Real} = reinterpret(T, Tensor.offset!(m, sizeof(T) * len))
@inline Tensor.offset!(m::Tensor.TensorMessageEncoder, offset::NTuple{N,Int}) where {N} = Tensor.offset!(Int32, m, length(offset)) .= offset

@inline Tensor.values(::Type{T}, m::Tensor.TensorMessageDecoder) where {T<:Real} = reinterpret(T, Tensor.values(m))
@inline Tensor.values!(::Type{T}, m::Tensor.TensorMessageEncoder, len) where {T<:Real} = reinterpret(T, Tensor.values!(m, sizeof(T) * len))

@inline function (m::Tensor.TensorMessageDecoder)()
    Tensor.sbe_rewind!(m)
    T = Base.eltype(Tensor.format(m))
    O = Tensor.order(m)
    N = div(Int(Tensor.dims_length(m)), sizeof(Int32))

    return values(AbstractTensorMessageArray{T,N,O}, m)
end
@inline function (m::Tensor.TensorMessageDecoder)(::Type{A}) where {T,N,A<:AbstractArray{T,N}}
    values(AbstractTensorMessageArray{T,N,order(A)}, m)
end
@inline function (m::Tensor.TensorMessageEncoder)(::Type{A}, dims; offset=nothing) where {T,N,A<:AbstractArray{T,N}}
    values!(AbstractTensorMessageArray{T,N,order(A)}, m, dims; offset)
end
@inline function (m::Tensor.TensorMessageEncoder)(::Type{A}, dims...; offset=nothing) where {T,N,A<:AbstractArray{T,N}}
    values!(AbstractTensorMessageArray{T,N,order(A)}, m, dims; offset)
end
@inline function (m::Tensor.TensorMessageEncoder)(src::A; offset=nothing) where {T,N,A<:AbstractArray{T,N}}
    values!(AbstractTensorMessageArray{T,N,order(A)}, m, src; offset)
end

function Base.show(io::IO, m::Tensor.TensorMessage{T}) where {T}
    println(io, "TensorMessage view over a type $T")
    println(io, "SbeBlockLength: ", Tensor.sbe_block_length(m))
    println(io, "SbeTemplateId:  ", Tensor.sbe_template_id(m))
    println(io, "SbeSchemaId:    ", Tensor.sbe_schema_id(m))
    println(io, "SbeSchemaVersion: ", Tensor.sbe_schema_version(m))

    writer = Tensor.TensorMessageDecoder(Tensor.sbe_buffer(m), Tensor.sbe_offset(m),
        Tensor.sbe_block_length(m), Tensor.sbe_schema_version(m))
    print(io, "header: ")
    Tensor.show(io, Tensor.header(writer))

    println(io)
    print(io, "format: ")
    print(io, Tensor.format(writer))

    println(io)
    print(io, "order: ")
    print(io, Tensor.order(writer))

    println(io)
    print(io, "dims: ")
    print(io, Tensor.dims(NTuple, writer))

    println(io)
    print(io, "offset: ")
    print(io, Tensor.offset(NTuple, writer))

    println(io)
    print(io, "values: ")
    print(io, Tensor.skip_values!(writer))
    print(io, " bytes of raw data")
end
