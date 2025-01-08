function Base.convert(::Type{UnsafeArray{UInt8}}, s::Symbol)
    p = Base.unsafe_convert(Ptr{UInt8}, s)
    len = @ccall strlen(p::Ptr{UInt8})::Csize_t
    UnsafeArray(p, (Int64(len),))
end