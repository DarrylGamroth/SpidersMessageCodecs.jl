# Generated SBE (Simple Binary Encoding) message codec
# Code generated by SBE. DO NOT EDIT.

struct SpidersMessageHeader{T<:AbstractArray{UInt8}}
    buffer::T
    offset::Int64
    acting_version::UInt16
    function SpidersMessageHeader(buffer::T, offset::Int64=0, acting_version::Integer=0) where {T}
        new{T}(buffer, offset, acting_version)
    end
end
const SpidersMessageHeaderDecoder = SpidersMessageHeader
const SpidersMessageHeaderEncoder = SpidersMessageHeader

sbe_buffer(m::SpidersMessageHeader) = m.buffer
sbe_offset(m::SpidersMessageHeader) = m.offset
sbe_acting_version(m::SpidersMessageHeader) = m.acting_version
sbe_encoded_length(::SpidersMessageHeader) = UInt16(0x40)
sbe_encoded_length(::Type{<:SpidersMessageHeader}) = UInt16(0x40)
sbe_schema_id(::SpidersMessageHeader) = UInt16(0x1)
sbe_schema_id(::Type{<:SpidersMessageHeader}) = UInt16(0x1)
sbe_schema_version(::SpidersMessageHeader) = UInt16(0x0)
sbe_schema_version(::Type{<:SpidersMessageHeader}) = UInt16(0x0)

function channelRcvTimestampNs_meta_attribute(::SpidersMessageHeader, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end
channelRcvTimestampNs_id(::SpidersMessageHeader) = UInt16(0xffffffffffffffff)
channelRcvTimestampNs_since_version(::SpidersMessageHeader) = UInt16(0x0)
channelRcvTimestampNs_in_acting_version(m::SpidersMessageHeader) = sbe_acting_version(m) >= UInt16(0x0)
channelRcvTimestampNs_encoding_offset(::SpidersMessageHeader) = 0
channelRcvTimestampNs_null_value(::SpidersMessageHeader) = Int64(-9223372036854775808)
channelRcvTimestampNs_min_value(::SpidersMessageHeader) = Int64(-9223372036854775807)
channelRcvTimestampNs_max_value(::SpidersMessageHeader) = Int64(9223372036854775807)
channelRcvTimestampNs_encoding_length(::SpidersMessageHeader) = 8

@inline function channelRcvTimestampNs(m::SpidersMessageHeaderDecoder)
    return decode_le(Int64, m.buffer, m.offset + 0)
end
@inline channelRcvTimestampNs!(m::SpidersMessageHeaderEncoder, value) = encode_le(Int64, m.buffer, m.offset + 0, value)

function channelSndTimestampNs_meta_attribute(::SpidersMessageHeader, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end
channelSndTimestampNs_id(::SpidersMessageHeader) = UInt16(0xffffffffffffffff)
channelSndTimestampNs_since_version(::SpidersMessageHeader) = UInt16(0x0)
channelSndTimestampNs_in_acting_version(m::SpidersMessageHeader) = sbe_acting_version(m) >= UInt16(0x0)
channelSndTimestampNs_encoding_offset(::SpidersMessageHeader) = 8
channelSndTimestampNs_null_value(::SpidersMessageHeader) = Int64(-9223372036854775808)
channelSndTimestampNs_min_value(::SpidersMessageHeader) = Int64(-9223372036854775807)
channelSndTimestampNs_max_value(::SpidersMessageHeader) = Int64(9223372036854775807)
channelSndTimestampNs_encoding_length(::SpidersMessageHeader) = 8

@inline function channelSndTimestampNs(m::SpidersMessageHeaderDecoder)
    return decode_le(Int64, m.buffer, m.offset + 8)
end
@inline channelSndTimestampNs!(m::SpidersMessageHeaderEncoder, value) = encode_le(Int64, m.buffer, m.offset + 8, value)

function timestampNs_meta_attribute(::SpidersMessageHeader, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end
timestampNs_id(::SpidersMessageHeader) = UInt16(0xffffffffffffffff)
timestampNs_since_version(::SpidersMessageHeader) = UInt16(0x0)
timestampNs_in_acting_version(m::SpidersMessageHeader) = sbe_acting_version(m) >= UInt16(0x0)
timestampNs_encoding_offset(::SpidersMessageHeader) = 16
timestampNs_null_value(::SpidersMessageHeader) = Int64(-9223372036854775808)
timestampNs_min_value(::SpidersMessageHeader) = Int64(-9223372036854775807)
timestampNs_max_value(::SpidersMessageHeader) = Int64(9223372036854775807)
timestampNs_encoding_length(::SpidersMessageHeader) = 8

@inline function timestampNs(m::SpidersMessageHeaderDecoder)
    return decode_le(Int64, m.buffer, m.offset + 16)
end
@inline timestampNs!(m::SpidersMessageHeaderEncoder, value) = encode_le(Int64, m.buffer, m.offset + 16, value)

function correlationId_meta_attribute(::SpidersMessageHeader, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end
correlationId_id(::SpidersMessageHeader) = UInt16(0xffffffffffffffff)
correlationId_since_version(::SpidersMessageHeader) = UInt16(0x0)
correlationId_in_acting_version(m::SpidersMessageHeader) = sbe_acting_version(m) >= UInt16(0x0)
correlationId_encoding_offset(::SpidersMessageHeader) = 24
correlationId_null_value(::SpidersMessageHeader) = Int64(-9223372036854775808)
correlationId_min_value(::SpidersMessageHeader) = Int64(-9223372036854775807)
correlationId_max_value(::SpidersMessageHeader) = Int64(9223372036854775807)
correlationId_encoding_length(::SpidersMessageHeader) = 8

@inline function correlationId(m::SpidersMessageHeaderDecoder)
    return decode_le(Int64, m.buffer, m.offset + 24)
end
@inline correlationId!(m::SpidersMessageHeaderEncoder, value) = encode_le(Int64, m.buffer, m.offset + 24, value)

function tag_meta_attribute(::SpidersMessageHeader, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end
tag_id(::SpidersMessageHeader) = UInt16(0xffffffffffffffff)
tag_since_version(::SpidersMessageHeader) = UInt16(0x0)
tag_in_acting_version(m::SpidersMessageHeader) = sbe_acting_version(m) >= UInt16(0x0)
tag_encoding_offset(::SpidersMessageHeader) = 32
tag_null_value(::SpidersMessageHeader) = UInt8(0x0)
tag_min_value(::SpidersMessageHeader) = UInt8(0x20)
tag_max_value(::SpidersMessageHeader) = UInt8(0x7e)
tag_encoding_length(::SpidersMessageHeader) = 32
tag_length(::SpidersMessageHeader) = 32
tag_eltype(::SpidersMessageHeader) = UInt8

@inline function tag(m::SpidersMessageHeaderDecoder)
    return mappedarray(ltoh, reinterpret(UInt8, view(m.buffer, m.offset+32+1:m.offset+32+sizeof(UInt8)*32)))
end

@inline function tag(::Type{<:SVector},m::SpidersMessageHeaderDecoder)
    return mappedarray(ltoh, reinterpret(SVector{32,UInt8}, view(m.buffer, m.offset+32+1:m.offset+32+sizeof(UInt8)*32))[])
end

@inline function tag(::Type{<:AbstractString}, m::SpidersMessageHeaderDecoder)
    value = view(m.buffer, m.offset+1+32:m.offset+32+sizeof(UInt8)*32)
    return StringView(rstrip_nul(value))
end

@inline function tag!(m::SpidersMessageHeaderEncoder)
    return mappedarray(ltoh, htol, reinterpret(UInt8, view(m.buffer, m.offset+32+1:m.offset+32+sizeof(UInt8)*32)))
end

@inline function tag!(m::SpidersMessageHeaderEncoder, value)
    copyto!(mappedarray(ltoh, htol, reinterpret(UInt8, view(m.buffer, m.offset+32+1:m.offset+32+sizeof(UInt8)*32))), value)
end

function show(io::IO, writer::SpidersMessageHeader{T}) where {T}
    println(io, "SpidersMessageHeader view over a type $T")
    print(io, "channelRcvTimestampNs: ")
    print(io, channelRcvTimestampNs(writer))

    println(io)
    print(io, "channelSndTimestampNs: ")
    print(io, channelSndTimestampNs(writer))

    println(io)
    print(io, "TimestampNs: ")
    print(io, timestampNs(writer))

    println(io)
    print(io, "correlationId: ")
    print(io, correlationId(writer))

    println(io)
    print(io, "tag: ")
    print(io, "\"")
    print(io, tag(StringView, writer))
    print(io, "\"")

end
