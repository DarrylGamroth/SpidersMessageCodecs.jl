module SpidersMessageCodecs

using LinearAlgebra
using StringViews
using UnsafeArrays
using ValSplit

function Base.convert(::Type{UnsafeArray{UInt8}}, s::Symbol)
    p = Base.unsafe_convert(Ptr{UInt8}, s)
    len = @ccall strlen(p::Ptr{UInt8})::Csize_t
    UnsafeArray(p, (Int64(len),))
end

# Include SBE generated code
include("Tensor.jl")

# Include Event last
include("Event.jl")

include("Sbe.jl")

end # module SpidersMessageCodecs