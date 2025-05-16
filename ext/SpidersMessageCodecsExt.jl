module SpidersMessageCodecsExt

using LinearAlgebra
using SpidersMessageCodecs
using StringViews
using UnsafeArrays
using ValSplit

abstract type SbeType end

@valsplit function message_type(Val(templateId::UInt16))
    throw(ArgumentError("Unknown message templateId=$templateId"))
end

# Type-unstable function to decode a SBE message
function SpidersMessageCodecs.decode(buffer::AbstractArray, offset::Integer=0;
    position_ptr::Ref{Int64}=Ref(0),
    header::MessageHeader=MessageHeader(buffer, offset))
    T = message_type(SpidersMessageCodecs.templateId(header))
    SpidersMessageCodecs.decode(T, buffer, offset; position_ptr, header)
end

include("format.jl")

include("EventMessage.jl")
include("TensorMessage.jl")

end # module SpidersMessageCodecsExt
