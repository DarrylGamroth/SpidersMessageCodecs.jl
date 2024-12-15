@valsplit function sbe_message_type(Val(templateId::UInt16), Val(schemaId::UInt16))
    throw(ArgumentError("Unknown message templateId=$templateId or schemaId=$schemaId"))
end

function sbe_decoder(buffer::AbstractArray, offset::Int, position_ptr::Base.RefValue)
    header = Spidersheader.MessageHeader(buffer, offset)
    templateId = Spidersheader.templateId(header)
    schemaId = Spidersheader.schemaId(header)
    T = sbe_message_type(templateId, schemaId)
    sbe_decoder(T, buffer, offset, position_ptr)
end

sbe_decoder(buffer::AbstractArray, position_ptr::Base.RefValue) = sbe_decoder(buffer, 0, position_ptr)
sbe_decoder(buffer::AbstractArray) = sbe_decoder(buffer, Ref(0))
sbe_decoder(::Type{T}, buffer::AbstractArray, position_ptr::Base.RefValue) where {T} = sbe_decoder(T, buffer, 0, position_ptr)
sbe_decoder(::Type{T}, buffer::AbstractArray) where {T} = sbe_decoder(T, buffer, Ref(0))

is_sbe_message(::Type{T}) where {T} = false
