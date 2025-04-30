message_type(::Val{sbe_template_id(TensorMessage)}) = TensorMessage

function SpidersMessageCodecs.decode(::Type{<:TensorMessage}, buffer::AbstractArray, args...; kwargs...)
    TensorMessageDecoder(buffer::AbstractArray, args...; kwargs...)
end
function SpidersMessageCodecs.encode(::Type{<:TensorMessage}, buffer::AbstractArray, args...; kwargs...)
    TensorMessageEncoder(buffer::AbstractArray, args...; kwargs...)
end

is_sbe_message(::Type{<:TensorMessage}) = true

order(::Type{<:Adjoint}) = SpidersMessageCodecs.MajorOrder.ROW
order(::Type{<:Transpose}) = SpidersMessageCodecs.MajorOrder.ROW
order(::Type{<:AbstractArray}) = SpidersMessageCodecs.MajorOrder.COLUMN

abstract type AbstractTensorMessageArray{T,N,O} end

# Type-stable
@inline function values(::Type{<:AbstractTensorMessageArray{T,N,O}}, m::TensorMessageDecoder) where {T,N,O}
    sbe_rewind!(m)

    # FIXME these checks slow down the code
    # type = Base.eltype(format(m))
    # T <: type || error("Unexpected data type, expected $T but got $type")

    # order = SpidersMessageCodecs.order(m)
    # O == order || error("Unexpected order, expected $O but got $order")

    dims = SpidersMessageCodecs.dims(NTuple{N}, m)
    SpidersMessageCodecs.skip_offset!(m)

    data = SpidersMessageCodecs.values(T, m)
    reshape(data, dims)
end

# Type-stable
@inline function values!(
    ::Type{<:AbstractTensorMessageArray{T,N,O}},
    m::TensorMessageEncoder,
    dims::NTuple{N,Int};
    offset::Union{Nothing,NTuple{N,Int}}=nothing) where {T,N,O}

    sbe_rewind!(m)

    SpidersMessageCodecs.format!(m, convert(SpidersMessageCodecs.Format.SbeEnum, T))
    SpidersMessageCodecs.order!(m, O)

    SpidersMessageCodecs.dims!(m, dims)
    SpidersMessageCodecs.offset!(m, offset)
    data = SpidersMessageCodecs.values!(T, m; length=prod(dims))
    reshape(data, dims)
end

@inline function values!(
    t::Type{<:AbstractTensorMessageArray{T,N,O}},
    m::TensorMessageEncoder,
    src::AbstractArray{T,N}; offset=nothing) where {T,N,O}
    values!(t, m, size(src); offset) .= src
end

@inline SpidersMessageCodecs.dims(::Type{T}, m::TensorMessageDecoder) where {T<:Real} = reinterpret(T, SpidersMessageCodecs.dims(m))
@inline SpidersMessageCodecs.dims(::Type{NTuple}, m::TensorMessageDecoder) = Tuple(SpidersMessageCodecs.dims(Int32, m))
@inline function SpidersMessageCodecs.dims(::Type{NTuple{N}}, m::TensorMessageDecoder) where {N}
    dims = SpidersMessageCodecs.dims(Int32, m)
    ntuple(i -> dims[i], Val(N))
end
@inline SpidersMessageCodecs.dims!(::Type{T}, m::TensorMessageEncoder; length::Int) where {T<:Real} = reinterpret(T, SpidersMessageCodecs.dims!(m; length=sizeof(T) * length))
@inline SpidersMessageCodecs.dims!(m::TensorMessageEncoder, dims::NTuple{N,Int}) where {N} = SpidersMessageCodecs.dims!(Int32, m; length=Base.length(dims)) .= dims

@inline SpidersMessageCodecs.offset(::Type{T}, m::TensorMessageDecoder) where {T<:Real} = reinterpret(T, SpidersMessageCodecs.offset(m))
@inline SpidersMessageCodecs.offset(::Type{NTuple}, m::TensorMessageDecoder) = Tuple(SpidersMessageCodecs.offset(Int32, m))
@inline function SpidersMessageCodecs.offset(::Type{NTuple{N}}, m::TensorMessageDecoder) where {N}
    offset = SpidersMessageCodecs.offset(Int32, m)
    ntuple(i -> offset[i], Val(N))
end
@inline SpidersMessageCodecs.offset!(::Type{T}, m::TensorMessageEncoder; length::Int) where {T<:Real} = reinterpret(T, SpidersMessageCodecs.offset!(m; length=sizeof(T) * length))
@inline SpidersMessageCodecs.offset!(m::TensorMessageEncoder, offset::NTuple{N,Int}) where {N} = SpidersMessageCodecs.offset!(Int32, m; length=Base.length(offset)) .= offset
@inline SpidersMessageCodecs.offset!(m::TensorMessageEncoder, ::Nothing) = SpidersMessageCodecs.offset!(m; length=0)

@inline SpidersMessageCodecs.values(::Type{T}, m::TensorMessageDecoder) where {T<:Real} = reinterpret(T, SpidersMessageCodecs.values(m))
@inline SpidersMessageCodecs.values!(::Type{T}, m::TensorMessageEncoder; length::Int) where {T<:Real} = reinterpret(T, SpidersMessageCodecs.values!(m; length=sizeof(T) * length))

@inline function SpidersMessageCodecs.decode(m::TensorMessageDecoder)
    SpidersMessageCodecs.sbe_rewind!(m)
    T = Base.eltype(SpidersMessageCodecs.format(m))
    O = SpidersMessageCodecs.order(m)
    N = div(Int(SpidersMessageCodecs.dims_length(m)), sizeof(Int32))
    return values(AbstractTensorMessageArray{T,N,O}, m)
end
@inline function SpidersMessageCodecs.decode(::Type{A}, m::TensorMessageDecoder) where {T,N,A<:AbstractArray{T,N}}
    values(AbstractTensorMessageArray{T,N,order(A)}, m)
end
@inline function SpidersMessageCodecs.encode(::Type{A}, m::TensorMessageEncoder, dims; offset=nothing) where {T,N,A<:AbstractArray{T,N}}
    values!(AbstractTensorMessageArray{T,N,order(A)}, m, dims; offset)
end
@inline function SpidersMessageCodecs.encode(::Type{A}, m::TensorMessageEncoder, dims...; offset=nothing) where {T,N,A<:AbstractArray{T,N}}
    values!(AbstractTensorMessageArray{T,N,order(A)}, m, dims; offset)
end
@inline function SpidersMessageCodecs.encode(m::TensorMessageEncoder, src::A; offset=nothing) where {T,N,A<:AbstractArray{T,N}}
    values!(AbstractTensorMessageArray{T,N,order(A)}, m, src; offset)
end

function Base.show(io::IO, m::TensorMessage{T}) where {T}
    println(io, "TensorMessage view over a type $T")
    println(io, "SbeBlockLength: ", sbe_block_length(m))
    println(io, "SbeTemplateId:  ", sbe_template_id(m))
    println(io, "SbeSchemaId:    ", sbe_schema_id(m))
    println(io, "SbeSchemaVersion: ", sbe_schema_version(m))

    writer = SpidersMessageCodecs.TensorMessageDecoder(sbe_buffer(m), sbe_offset(m), Ref(0),
        sbe_block_length(m), sbe_schema_version(m))
    print(io, "header: ")
    SpidersMessageCodecs.show(io, SpidersMessageCodecs.header(writer))

    println(io)
    print(io, "format: ")
    print(io, SpidersMessageCodecs.format(writer))

    println(io)
    print(io, "order: ")
    print(io, SpidersMessageCodecs.order(writer))

    println(io)
    print(io, "dims: ")
    print(io, SpidersMessageCodecs.dims(NTuple, writer))

    println(io)
    print(io, "offset: ")
    print(io, SpidersMessageCodecs.offset(NTuple, writer))

    println(io)
    print(io, "values: ")
    print(io, SpidersMessageCodecs.skip_values!(writer))
    print(io, " bytes of raw data")
end
