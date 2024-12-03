# Generated SBE (Simple Binary Encoding) message codec
# Code generated by SBE. DO NOT EDIT.

struct MessageHeader{T<:AbstractArray{UInt8}}
    buffer::T
    offset::Int64
    acting_version::Int64
    function MessageHeader(buffer::T, offset=0, acting_version=0) where {T}
        checkbounds(buffer, offset + 1 + 8)
        new{T}(buffer, offset, acting_version)
    end
end

MessageHeader() = MessageHeader(UInt8[])

sbe_buffer(m::MessageHeader) = @inbounds view(m.buffer, m.offset+1:m.offset+8)
sbe_offset(m::MessageHeader) = m.offset
sbe_acting_version(m::MessageHeader) = m.acting_version
sbe_encoded_length(::MessageHeader) = 8
sbe_schema_id(::MessageHeader) = 6
sbe_schema_version(::MessageHeader) = 0

function blockLength_meta_attribute(::MessageHeader, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    error(lazy"unknown attribute: $meta_attribute")
end
blockLength_id(::MessageHeader) = -1
blockLength_in_acting_version(m::MessageHeader) = sbe_acting_version(m) >= 0
blockLength_encoding_offset(::MessageHeader) = 0
blockLength_null_value(::MessageHeader) = UInt16(0xffff)
blockLength_min_value(::MessageHeader) = UInt16(0x0)
blockLength_max_value(::MessageHeader) = UInt16(0xfffe)
blockLength_encoding_length(::MessageHeader) = 2

@inline function blockLength(m::MessageHeader)
    return decode_le(UInt16, m.buffer, m.offset + 0)
end
@inline blockLength!(m::MessageHeader, value) = encode_le(UInt16, m.buffer, m.offset + 0, value)

function templateId_meta_attribute(::MessageHeader, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    error(lazy"unknown attribute: $meta_attribute")
end
templateId_id(::MessageHeader) = -1
templateId_in_acting_version(m::MessageHeader) = sbe_acting_version(m) >= 0
templateId_encoding_offset(::MessageHeader) = 2
templateId_null_value(::MessageHeader) = UInt16(0xffff)
templateId_min_value(::MessageHeader) = UInt16(0x0)
templateId_max_value(::MessageHeader) = UInt16(0xfffe)
templateId_encoding_length(::MessageHeader) = 2

@inline function templateId(m::MessageHeader)
    return decode_le(UInt16, m.buffer, m.offset + 2)
end
@inline templateId!(m::MessageHeader, value) = encode_le(UInt16, m.buffer, m.offset + 2, value)

function schemaId_meta_attribute(::MessageHeader, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    error(lazy"unknown attribute: $meta_attribute")
end
schemaId_id(::MessageHeader) = -1
schemaId_in_acting_version(m::MessageHeader) = sbe_acting_version(m) >= 0
schemaId_encoding_offset(::MessageHeader) = 4
schemaId_null_value(::MessageHeader) = UInt16(0xffff)
schemaId_min_value(::MessageHeader) = UInt16(0x0)
schemaId_max_value(::MessageHeader) = UInt16(0xfffe)
schemaId_encoding_length(::MessageHeader) = 2

@inline function schemaId(m::MessageHeader)
    return decode_le(UInt16, m.buffer, m.offset + 4)
end
@inline schemaId!(m::MessageHeader, value) = encode_le(UInt16, m.buffer, m.offset + 4, value)

function version_meta_attribute(::MessageHeader, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    error(lazy"unknown attribute: $meta_attribute")
end
version_id(::MessageHeader) = -1
version_in_acting_version(m::MessageHeader) = sbe_acting_version(m) >= 0
version_encoding_offset(::MessageHeader) = 6
version_null_value(::MessageHeader) = UInt16(0xffff)
version_min_value(::MessageHeader) = UInt16(0x0)
version_max_value(::MessageHeader) = UInt16(0xfffe)
version_encoding_length(::MessageHeader) = 2

@inline function version(m::MessageHeader)
    return decode_le(UInt16, m.buffer, m.offset + 6)
end
@inline version!(m::MessageHeader, value) = encode_le(UInt16, m.buffer, m.offset + 6, value)

function print_fields(io::IO, writer::MessageHeader{T}) where {T}
    println(io, "MessageHeader view over a type $T")
    print(io, "blockLength: ")
    print(io, blockLength(writer))

    println(io)
    print(io, "templateId: ")
    print(io, templateId(writer))

    println(io)
    print(io, "schemaId: ")
    print(io, schemaId(writer))

    println(io)
    print(io, "version: ")
    print(io, version(writer))

end
