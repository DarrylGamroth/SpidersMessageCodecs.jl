# Generated SBE (Simple Binary Encoding) message codec
# Code generated by SBE. DO NOT EDIT.

struct VarDataEncoding{T<:AbstractArray{UInt8}}
    buffer::T
    offset::Int64
    acting_version::UInt16
    function VarDataEncoding(buffer::T, offset=0, acting_version=0) where {T}
        new{T}(buffer, offset, acting_version)
    end
end
const VarDataEncodingDecoder = VarDataEncoding
const VarDataEncodingEncoder = VarDataEncoding

sbe_buffer(m::VarDataEncoding) = m.buffer
sbe_offset(m::VarDataEncoding) = m.offset
sbe_acting_version(m::VarDataEncoding) = m.acting_version
sbe_encoded_length(::VarDataEncoding) = typemax(UInt16)
sbe_encoded_length(::Type{<:VarDataEncoding}) = typemax(UInt16)
sbe_schema_id(::VarDataEncoding) = UInt16(0x6)
sbe_schema_id(::Type{<:VarDataEncoding}) = UInt16(0x6)
sbe_schema_version(::VarDataEncoding) = UInt16(0x0)
sbe_schema_version(::Type{<:VarDataEncoding}) = UInt16(0x0)

function length_meta_attribute(::VarDataEncoding, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end
length_id(::VarDataEncoding) = UInt16(0xffffffffffffffff)
length_since_version(::VarDataEncoding) = UInt16(0x0)
length_in_acting_version(m::VarDataEncoding) = sbe_acting_version(m) >= UInt16(0x0)
length_encoding_offset(::VarDataEncoding) = 0
length_null_value(::VarDataEncoding) = UInt32(0xffffffff)
length_min_value(::VarDataEncoding) = UInt32(0x0)
length_max_value(::VarDataEncoding) = UInt32(0x40000000)
length_encoding_length(::VarDataEncoding) = 4

@inline function length(m::VarDataEncodingDecoder)
    return decode_le(UInt32, m.buffer, m.offset + 0)
end
@inline length!(m::VarDataEncodingEncoder, value) = encode_le(UInt32, m.buffer, m.offset + 0, value)

function varData_meta_attribute(::VarDataEncoding, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end
varData_id(::VarDataEncoding) = UInt16(0xffffffffffffffff)
varData_since_version(::VarDataEncoding) = UInt16(0x0)
varData_in_acting_version(m::VarDataEncoding) = sbe_acting_version(m) >= UInt16(0x0)
varData_encoding_offset(::VarDataEncoding) = 4
varData_null_value(::VarDataEncoding) = UInt8(0xff)
varData_min_value(::VarDataEncoding) = UInt8(0x0)
varData_max_value(::VarDataEncoding) = UInt8(0xfe)
varData_encoding_length(::VarDataEncoding) = -1

function show(io::IO, writer::VarDataEncoding{T}) where {T}
    println(io, "VarDataEncoding view over a type $T")
    print(io, "length: ")
    print(io, length(writer))

end
