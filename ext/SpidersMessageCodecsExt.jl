module SpidersMessageCodecsExt

using LinearAlgebra
using SpidersMessageCodecs
using StringViews
using SymbolConverters
using UnsafeArrays
using ValSplit

@valsplit function message_type(Val(templateId::UInt16))
    throw(ArgumentError("Unknown message templateId=$templateId"))
end

# Type-unstable function to decode a SBE message
function SpidersMessageCodecs.decode(buffer::AbstractArray, offset::Int64=0; position_ptr::Base.RefValue=Ref(0))
    header = MessageHeader(buffer, offset)
    T = message_type(SpidersMessageCodecs.templateId(header))
    SpidersMessageCodecs.decode(T, buffer, offset; position_ptr)
end

function Base.convert(::Type{<:AbstractArray{UInt8}}, m)
    offset = sbe_offset(m) - sbe_encoded_length(MessageHeader)
    offset < 0 && throw(ArgumentError("Message offset is negative"))
    len = sbe_decoded_length(m)
    return view(sbe_buffer(m), offset+1:sbe_offset(m)+len)
end

abstract type SbeType end

is_sbe_message(::Type{T}) where {T} = false

include("format.jl")

include("EventMessage.jl")
include("TensorMessage.jl")

end # module SpidersMessageCodecsExt