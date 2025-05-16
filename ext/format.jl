function Base.eltype(T::SpidersMessageCodecs.Format.SbeEnum)
    T == SpidersMessageCodecs.Format.NOTHING ? Nothing :
    T == SpidersMessageCodecs.Format.UINT8 ? UInt8 :
    T == SpidersMessageCodecs.Format.INT8 ? Int8 :
    T == SpidersMessageCodecs.Format.UINT16 ? UInt16 :
    T == SpidersMessageCodecs.Format.INT16 ? Int16 :
    T == SpidersMessageCodecs.Format.UINT32 ? UInt32 :
    T == SpidersMessageCodecs.Format.INT32 ? Int32 :
    T == SpidersMessageCodecs.Format.UINT64 ? UInt64 :
    T == SpidersMessageCodecs.Format.INT64 ? Int64 :
    T == SpidersMessageCodecs.Format.FLOAT32 ? Float32 :
    T == SpidersMessageCodecs.Format.FLOAT64 ? Float64 :
    T == SpidersMessageCodecs.Format.BOOLEAN ? Bool :
    T == SpidersMessageCodecs.Format.STRING ? AbstractString :
    T == SpidersMessageCodecs.Format.BYTES ? AbstractVector{UInt8} :
    T == SpidersMessageCodecs.Format.BIT ? BitArray :
    T == SpidersMessageCodecs.Format.SBE ? SbeType :
    throw(ArgumentError("unexpected format"))
end

Base.convert(::Type{SpidersMessageCodecs.Format.SbeEnum}, ::Type{Nothing}) = SpidersMessageCodecs.Format.NOTHING
Base.convert(::Type{SpidersMessageCodecs.Format.SbeEnum}, ::Type{UInt8}) = SpidersMessageCodecs.Format.UINT8
Base.convert(::Type{SpidersMessageCodecs.Format.SbeEnum}, ::Type{Int8}) = SpidersMessageCodecs.Format.INT8
Base.convert(::Type{SpidersMessageCodecs.Format.SbeEnum}, ::Type{UInt16}) = SpidersMessageCodecs.Format.UINT16
Base.convert(::Type{SpidersMessageCodecs.Format.SbeEnum}, ::Type{Int16}) = SpidersMessageCodecs.Format.INT16
Base.convert(::Type{SpidersMessageCodecs.Format.SbeEnum}, ::Type{UInt32}) = SpidersMessageCodecs.Format.UINT32
Base.convert(::Type{SpidersMessageCodecs.Format.SbeEnum}, ::Type{Int32}) = SpidersMessageCodecs.Format.INT32
Base.convert(::Type{SpidersMessageCodecs.Format.SbeEnum}, ::Type{UInt64}) = SpidersMessageCodecs.Format.UINT64
Base.convert(::Type{SpidersMessageCodecs.Format.SbeEnum}, ::Type{Int64}) = SpidersMessageCodecs.Format.INT64
Base.convert(::Type{SpidersMessageCodecs.Format.SbeEnum}, ::Type{Float32}) = SpidersMessageCodecs.Format.FLOAT32
Base.convert(::Type{SpidersMessageCodecs.Format.SbeEnum}, ::Type{Float64}) = SpidersMessageCodecs.Format.FLOAT64
Base.convert(::Type{SpidersMessageCodecs.Format.SbeEnum}, ::Type{Bool}) = SpidersMessageCodecs.Format.BOOLEAN
Base.convert(::Type{SpidersMessageCodecs.Format.SbeEnum}, ::Type{<:AbstractString}) = SpidersMessageCodecs.Format.STRING
Base.convert(::Type{SpidersMessageCodecs.Format.SbeEnum}, ::Type{<:AbstractVector{UInt8}}) = SpidersMessageCodecs.Format.BYTES
Base.convert(::Type{SpidersMessageCodecs.Format.SbeEnum}, ::Type{<:BitArray}) = SpidersMessageCodecs.Format.BIT
Base.convert(::Type{SpidersMessageCodecs.Format.SbeEnum}, ::Type{Symbol}) = SpidersMessageCodecs.Format.STRING
Base.convert(::Type{SpidersMessageCodecs.Format.SbeEnum}, ::Type{<:Enum}) = SpidersMessageCodecs.Format.INT64
