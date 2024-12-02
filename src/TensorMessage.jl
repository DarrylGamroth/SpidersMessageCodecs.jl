include("../gen/tensor/Tensor.jl")

# FIXME Rename values in the SBE schema to array, or add an methods using the Semantic Type field
# Just empty ones like Tensor."semantic_type" so it can be overridden 

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
    throw(ArgumentError("unexpected type"))
end

abstract type AbstractTensorMessageArray{T,N,O} end

@inline function Tensor.shape(::Type{NTuple{N}}, m::Tensor.TensorMessage) where {N}
    shape = Tensor.shape(Int32, m)
    ntuple(i -> shape[i], Val(N))
end

@inline function Tensor.shape(::Type{Tuple}, m::Tensor.TensorMessage)
    Tuple(Tensor.shape(Int32, m))
end

@inline function Tensor.shape!(m::Tensor.TensorMessage, shape::NTuple{N,Int}) where {N}
    Tensor.shape!(Int32, m, length(shape)) .= shape
end

@inline function Tensor.offset(::Type{NTuple{N,T}}, m::Tensor.TensorMessage) where {N,T}
    offset = Tensor.offset(Int32, m)
    ntuple(i -> offset[i], Val(N))
end

@inline function Tensor.offset(::Type{Tuple}, m::Tensor.TensorMessage)
    Tuple(Tensor.offset(Int32, m))
end

@inline function Tensor.offset!(m::Tensor.TensorMessage, offset::NTuple{N,Int}) where {N}
    Tensor.offset!(Int32, m, length(offset)) .= offset
end

@inline function Tensor.values(::Type{<:AbstractTensorMessageArray{T,N,O}}, m::Tensor.TensorMessage) where {T,N,O}
    Tensor.sbe_rewind!(m)

    type = Base.eltype(Tensor.format(m))
    T == type || error("Unexpected data type, expected $T but got $type")

    order = Tensor.order(m)
    O == order || error("Unexpected order, expected $O but got $order")

    shape = Tensor.shape(NTuple{N}, m)

    # Skip offset for now
    Tensor.skip_offset!(m)

    data = Tensor.values(T, m)
    array = reshape(data, shape)

    return O == Tensor.Order.COLUMN ? array : array'
end

function Tensor.values(::Type{<:AbstractTensorMessageArray}, m::Tensor.TensorMessage)
    Tensor.sbe_rewind!(m)
    T = Base.eltype(Tensor.format(m))
    O = Tensor.order(m)
    N = div(Int(Tensor.shape_length(m)), sizeof(Int32))

    return Tensor.values(AbstractTensorMessageArray{T,N,O}, m)
end

# function Tensor.values_copyto!(dest::AbstractArray{T,N}, m::Tensor.TensorMessage)
#     copyto!(dest, Tensor.values(AbstractTensorMessageArray{T,N,Tensor.Order.COLUMN}, m))
# end

@inline function Tensor.values!(::Type{<:AbstractTensorMessageArray{T,N,O}}, m::Tensor.TensorMessage, shape::NTuple{N,Int}) where {T,N,O}
    Tensor.sbe_rewind!(m)
    Tensor.format!(m, inv_eltype(T))
    Tensor.order!(m, O)
    Tensor.shape!(m, shape)
    Tensor.skip_offset!(m)
    data = Tensor.values!(T, m, prod(shape))
    array = reshape(data, shape)

    return O == Tensor.Order.COLUMN ? array : array'
end

function Tensor.values!(t::Type{<:AbstractTensorMessageArray{T,N,O}}, m::Tensor.TensorMessage, src::AbstractArray{T,N}) where {T,N,O}
    Tensor.values!(t, m, size(src)) .= src
end

function Tensor.values!(m::Tensor.TensorMessage, src::AbstractArray{T,N}) where {T,N}
    Tensor.values!(AbstractTensorMessageArray{T,N,Tensor.Order.COLUMN}, m, src)
end

function Tensor.values!(m::Tensor.TensorMessage, src::Adjoint{T,A}) where {T,N,A<:AbstractArray{T,N}}
    Tensor.values!(AbstractTensorMessageArray{T,N,Tensor.Order.ROW}, m, src)
end

