module Sbe

using ValSplit

include("Utils.jl")
include("MessageHeader.jl")

# Return the SBE message type for the given templateId and schemaId
@valsplit function message_type(Val(templateId::UInt16), Val(schemaId::UInt16))
    throw(ArgumentError("Unknown message templateId=$templateId or schemaId=$schemaId"))
end

# Type-unstable function to decode a SBE message
function decoder(buffer::AbstractArray, offset::Int64, position_ptr::Base.RefValue)
    header = MessageHeader(buffer, offset)
    T = message_type(templateId(header), schemaId(header))
    decoder(T, buffer, offset, position_ptr)
end

decoder(buffer::AbstractArray, position_ptr::Base.RefValue) = decoder(buffer, 0, position_ptr)
decoder(buffer::AbstractArray) = decoder(buffer, Ref(0))
decoder(::Type{T}, buffer::AbstractArray, offset::Int64, position_ptr::Base.RefValue) where {T} = nothing
decoder(::Type{T}, buffer::AbstractArray, position_ptr::Base.RefValue) where {T} = decoder(T, buffer, 0, position_ptr)
decoder(::Type{T}, buffer::AbstractArray) where {T} = decoder(T, buffer, Ref(0))

is_sbe_message(::Type{T}) where {T} = false

end # module Sbe