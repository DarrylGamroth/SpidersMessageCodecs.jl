# Generated SBE (Simple Binary Encoding) message codec
# Code generated by SBE. DO NOT EDIT.

struct VarDataEncoding{T<:AbstractArray{UInt8}}
    buffer::T
    offset::Int64
    acting_version::Int64
    function VarDataEncoding(buffer::T, offset=0, acting_version=0) where {T}
        checkbounds(buffer, offset + 1 + -1)
        new{T}(buffer, offset, acting_version)
    end
end

VarDataEncoding() = VarDataEncoding(UInt8[])

sbe_buffer(m::VarDataEncoding) = @inbounds view(m.buffer, m.offset+1:m.offset+-1)
sbe_offset(m::VarDataEncoding) = m.offset
sbe_acting_version(m::VarDataEncoding) = m.acting_version
sbe_encoded_length(::VarDataEncoding) = typemax(UInt64)
sbe_schema_id(::VarDataEncoding) = 1
sbe_schema_version(::VarDataEncoding) = 0

function length_meta_attribute(::VarDataEncoding, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    error(lazy"unknown attribute: $meta_attribute")
end
length_id(::VarDataEncoding) = -1
length_in_acting_version(m::VarDataEncoding) = sbe_acting_version(m) >= 0
length_encoding_offset(::VarDataEncoding) = 0
length_null_value(::VarDataEncoding) = UInt32(0xffffffff)
length_min_value(::VarDataEncoding) = UInt32(0x0)
length_max_value(::VarDataEncoding) = UInt32(0x40000000)
length_encoding_length(::VarDataEncoding) = 4

@inline function length(m::VarDataEncoding)
    return decode_le(UInt32, m.buffer, m.offset + 0)
end
@inline length!(m::VarDataEncoding, value) = encode_le(UInt32, m.buffer, m.offset + 0, value)

function varData_meta_attribute(::VarDataEncoding, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    error(lazy"unknown attribute: $meta_attribute")
end
varData_id(::VarDataEncoding) = -1
varData_in_acting_version(m::VarDataEncoding) = sbe_acting_version(m) >= 0
varData_encoding_offset(::VarDataEncoding) = 4
varData_null_value(::VarDataEncoding) = UInt8(0xff)
varData_min_value(::VarDataEncoding) = UInt8(0x0)
varData_max_value(::VarDataEncoding) = UInt8(0xfe)
varData_encoding_length(::VarDataEncoding) = -1

function print_fields(io::IO, writer::VarDataEncoding{T}) where {T}
    println(io, "VarDataEncoding view over a type $T")
    print(io, "length: ")
    print(io, length(writer))

end
