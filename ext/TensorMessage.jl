message_type(::Val{sbe_template_id(TensorMessage)}) = TensorMessage

@inline function SpidersMessageCodecs.decode(::Type{<:TensorMessage}, buffer::AbstractArray, args...; kwargs...)
    TensorMessageDecoder(buffer::AbstractArray, args...; kwargs...)
end
@inline function SpidersMessageCodecs.encode(::Type{<:TensorMessage}, buffer::AbstractArray, args...; kwargs...)
    TensorMessageEncoder(buffer::AbstractArray, args...; kwargs...)
end

Base.convert(::Type{SpidersMessageCodecs.Format.SbeEnum}, ::Type{<:SpidersMessageCodecs.TensorMessage}) = SpidersMessageCodecs.Format.SBE

order(::Type{<:Adjoint}) = SpidersMessageCodecs.MajorOrder.ROW
order(::Type{<:Transpose}) = SpidersMessageCodecs.MajorOrder.ROW
order(::Type{<:AbstractArray}) = SpidersMessageCodecs.MajorOrder.COLUMN

abstract type AbstractTensorMessageArray{T,N,O} end

# Type-stable
@inline function _decode(::Type{<:AbstractTensorMessageArray{T,N,O}}, m::TensorMessageDecoder) where {T,N,O}
    sbe_rewind!(m)

    # FIXME these checks slow down the code
    # type = Base.eltype(format(m))
    # T <: type || error("Unexpected data type, expected $T but got $type")

    # majorOrder = SpidersMessageCodecs.majorOrder(m)
    # O == majorOrder || error("Unexpected majorOrder, expected $O but got $majorOrder")

    dims = SpidersMessageCodecs.dims(m, NTuple{N,Int32})
    SpidersMessageCodecs.skip_origin!(m)

    data = SpidersMessageCodecs.values(m, AbstractArray{T})
    reshape(data, dims)
end

# Type-stable
@inline function _encode(
    ::Type{<:AbstractTensorMessageArray{T,N,O}},
    m::TensorMessageEncoder,
    dims::NTuple{N};
    origin::Union{Nothing,NTuple{N}}=nothing) where {T,N,O}

    sbe_rewind!(m)

    SpidersMessageCodecs.format!(m, convert(SpidersMessageCodecs.Format.SbeEnum, T))
    SpidersMessageCodecs.majorOrder!(m, O)

    SpidersMessageCodecs.dims!(m, Int32.(dims))
    SpidersMessageCodecs.origin!(m, origin===nothing ? nothing : Int32.(origin))
    data = reinterpret(T, SpidersMessageCodecs.values_buffer!(m, prod(dims)*sizeof(T)))
    reshape(data, dims)
end

@inline function _encode(
    t::Type{<:AbstractTensorMessageArray{T,N,O}},
    m::TensorMessageEncoder,
    src::AbstractArray{T,N}; origin=nothing) where {T,N,O}
    _encode(t, m, size(src); origin) .= src
end

@inline function SpidersMessageCodecs.decode(m::TensorMessageDecoder)
    SpidersMessageCodecs.sbe_rewind!(m)
    T = Base.eltype(SpidersMessageCodecs.format(m))
    O = SpidersMessageCodecs.majorOrder(m)
    N = div(Int(SpidersMessageCodecs.dims_length(m)), sizeof(Int32))
    return _decode(AbstractTensorMessageArray{T,N,O}, m)
end
@inline function SpidersMessageCodecs.decode(m::TensorMessageDecoder, ::Type{A}) where {T,N,A<:AbstractArray{T,N}}
    _decode(AbstractTensorMessageArray{T,N,order(A)}, m)
end
@inline function SpidersMessageCodecs.encode(m::TensorMessageEncoder, ::Type{A}, dims; origin=nothing) where {T,N,A<:AbstractArray{T,N}}
    _encode(AbstractTensorMessageArray{T,N,order(A)}, m, dims; origin=origin)
end
@inline function SpidersMessageCodecs.encode(m::TensorMessageEncoder, ::Type{A}, dims...; origin=nothing) where {T,N,A<:AbstractArray{T,N}}
    _encode(AbstractTensorMessageArray{T,N,order(A)}, m, dims; origin=origin)
end
@inline function SpidersMessageCodecs.encode(m::TensorMessageEncoder, src::A; origin=nothing) where {T,N,A<:AbstractArray{T,N}}
    _encode(AbstractTensorMessageArray{T,N,order(A)}, m, src; origin=origin)
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
    print(io, "majorOrder: ")
    print(io, SpidersMessageCodecs.majorOrder(writer))

    println(io)
    print(io, "dims: ")
    print(io, Tuple(SpidersMessageCodecs.dims(writer, AbstractArray{Int32})))

    println(io)
    print(io, "origin: ")
    print(io, Tuple(SpidersMessageCodecs.origin(writer, AbstractArray{Int32})))

    println(io)
    print(io, "values: ")
    print(io, SpidersMessageCodecs.skip_values!(writer))
    print(io, " bytes of raw data")
end
