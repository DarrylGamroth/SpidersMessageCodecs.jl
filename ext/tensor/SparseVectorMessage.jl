# Generated SBE (Simple Binary Encoding) message codec
# Code generated by SBE. DO NOT EDIT.

abstract type SparseVectorMessage{T} end

struct SparseVectorMessageDecoder{T<:AbstractArray{UInt8}} <: SparseVectorMessage{T}
    buffer::T
    offset::Int64
    position_ptr::Base.RefValue{Int64}
    acting_block_length::UInt16
    acting_version::UInt16
    function SparseVectorMessageDecoder(buffer::T, offset::Int, position_ptr::Base.RefValue{Int64},
        acting_block_length::UInt16, acting_version::UInt16) where {T}
        position_ptr[] = offset + acting_block_length
        new{T}(buffer, offset, position_ptr, acting_block_length, acting_version)
    end
end

struct SparseVectorMessageEncoder{T<:AbstractArray{UInt8}} <: SparseVectorMessage{T}
    buffer::T
    offset::Int64
    position_ptr::Base.RefValue{Int64}
    function SparseVectorMessageEncoder(buffer::T, offset::Int, position_ptr::Base.RefValue{Int64}) where {T}
        position_ptr[] = offset + 76
        new{T}(buffer, offset, position_ptr)
    end
end

function SparseVectorMessageDecoder(buffer::AbstractArray, offset::Int, position_ptr::Base.RefValue{Int64},
    hdr::MessageHeader)
    if templateId(hdr) != UInt16(0x3) || schemaId(hdr) != UInt16(0x1)
        error("Template id or schema id mismatch")
    end
    SparseVectorMessageDecoder(buffer, offset + sbe_encoded_length(hdr), position_ptr,
        blockLength(hdr), version(hdr))
end
function SparseVectorMessageDecoder(buffer::AbstractArray, position_ptr::Base.RefValue{Int64},
    hdr::MessageHeader)
    SparseVectorMessageDecoder(buffer, 0, position_ptr, hdr)
end
function SparseVectorMessageDecoder(buffer::AbstractArray, offset::Int,
    acting_block_length::UInt16, acting_version::UInt16)
    SparseVectorMessageDecoder(buffer, offset, Ref(0), acting_block_length, acting_version)
end
function SparseVectorMessageDecoder(buffer::AbstractArray, offset::Int, hdr::MessageHeader)
    SparseVectorMessageDecoder(buffer, offset, Ref(0), hdr)
end
SparseVectorMessageDecoder(buffer::AbstractArray, hdr::MessageHeader) = SparseVectorMessageDecoder(buffer, 0, Ref(0), hdr)
function SparseVectorMessageEncoder(buffer::AbstractArray, position_ptr::Base.RefValue{Int64})
    SparseVectorMessageEncoder(buffer, 0, position_ptr)
end
function SparseVectorMessageEncoder(buffer::AbstractArray, offset::Int, position_ptr::Base.RefValue{Int64},
    hdr::MessageHeader)
    blockLength!(hdr, UInt16(0x4c))
    templateId!(hdr, UInt16(0x3))
    schemaId!(hdr, UInt16(0x1))
    version!(hdr, UInt16(0x0))
    SparseVectorMessageEncoder(buffer, offset + sbe_encoded_length(hdr), position_ptr)
end
function SparseVectorMessageEncoder(buffer::AbstractArray, position_ptr::Base.RefValue{Int64}, hdr::MessageHeader)
    SparseVectorMessageEncoder(buffer, 0, position_ptr, hdr)
end
function SparseVectorMessageEncoder(buffer::AbstractArray, offset::Int, hdr::MessageHeader)
    SparseVectorMessageEncoder(buffer, offset, Ref(0), hdr)
end
function SparseVectorMessageEncoder(buffer::AbstractArray, hdr::MessageHeader)
    SparseVectorMessageEncoder(buffer, 0, Ref(0), hdr)
end
SparseVectorMessageEncoder(buffer::AbstractArray, offset::Int=0) = SparseVectorMessageEncoder(buffer, offset, Ref(0))
sbe_buffer(m::SparseVectorMessage) = m.buffer
sbe_offset(m::SparseVectorMessage) = m.offset
sbe_position_ptr(m::SparseVectorMessage) = m.position_ptr
sbe_position(m::SparseVectorMessage) = m.position_ptr[]
sbe_position!(m::SparseVectorMessage, position) = m.position_ptr[] = position
sbe_block_length(::SparseVectorMessage) = UInt16(0x4c)
sbe_block_length(::Type{<:SparseVectorMessage}) = UInt16(0x4c)
sbe_template_id(::SparseVectorMessage) = UInt16(0x3)
sbe_template_id(::Type{<:SparseVectorMessage})  = UInt16(0x3)
sbe_schema_id(::SparseVectorMessage) = UInt16(0x1)
sbe_schema_id(::Type{<:SparseVectorMessage})  = UInt16(0x1)
sbe_schema_version(::SparseVectorMessage) = UInt16(0x0)
sbe_schema_version(::Type{<:SparseVectorMessage})  = UInt16(0x0)
sbe_semantic_type(::SparseVectorMessage) = ""
sbe_semantic_version(::SparseVectorMessage) = ""
sbe_acting_block_length(m::SparseVectorMessageDecoder) = m.acting_block_length
sbe_acting_block_length(::SparseVectorMessageEncoder) = UInt16(0x4c)
sbe_acting_version(m::SparseVectorMessageDecoder) = m.acting_version
sbe_acting_version(::SparseVectorMessageEncoder) = UInt16(0x0)
sbe_rewind!(m::SparseVectorMessage) = sbe_position!(m, m.offset + m.acting_block_length)
sbe_rewind!(m::SparseVectorMessageEncoder) = sbe_position!(m, m.offset + UInt16(0x4c))
sbe_encoded_length(m::SparseVectorMessage) = sbe_position(m) - m.offset
@inline function sbe_decoded_length(m::SparseVectorMessage)
    skipper = SparseVectorMessageDecoder(sbe_buffer(m), sbe_offset(m),
        sbe_acting_block_length(m), sbe_acting_version(m))
    sbe_rewind!(skipper)
    skip!(skipper)
    sbe_encoded_length(skipper)
end

function header_meta_attribute(::SparseVectorMessage, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end
header_id(::SparseVectorMessage) = UInt16(0x1)
header_since_version(::SparseVectorMessage) = UInt16(0x0)
header_in_acting_version(m::SparseVectorMessage) = sbe_acting_version(m) >= UInt16(0x0)
header_encoding_offset(::SparseVectorMessage) = 0
header(m::SparseVectorMessage) = SpidersMessageHeader(m.buffer, m.offset + 0, sbe_acting_version(m))

function format_meta_attribute(::SparseVectorMessage, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end
format_id(::SparseVectorMessage) = UInt16(0x2)
format_since_version(::SparseVectorMessage) = UInt16(0x0)
format_in_acting_version(m::SparseVectorMessage) = sbe_acting_version(m) >= UInt16(0x0)
format_encoding_offset(::SparseVectorMessage) = 64
format_encoding_length(::SparseVectorMessage) = 1
@inline function format(::Type{Integer}, m::SparseVectorMessageDecoder)
    return decode_le(Int8, m.buffer, m.offset + 64)
end
@inline function format(m::SparseVectorMessageDecoder)
    return Format.SbeEnum(decode_le(Int8, m.buffer, m.offset + 64))
end
@inline format!(m::SparseVectorMessageEncoder, value::Format.SbeEnum) = encode_le(Int8, m.buffer, m.offset + 64, Int8(value))

function indiciesFormat_meta_attribute(::SparseVectorMessage, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end
indiciesFormat_id(::SparseVectorMessage) = UInt16(0x3)
indiciesFormat_since_version(::SparseVectorMessage) = UInt16(0x0)
indiciesFormat_in_acting_version(m::SparseVectorMessage) = sbe_acting_version(m) >= UInt16(0x0)
indiciesFormat_encoding_offset(::SparseVectorMessage) = 65
indiciesFormat_encoding_length(::SparseVectorMessage) = 1
@inline function indiciesFormat(::Type{Integer}, m::SparseVectorMessageDecoder)
    return decode_le(Int8, m.buffer, m.offset + 65)
end
@inline function indiciesFormat(m::SparseVectorMessageDecoder)
    return Format.SbeEnum(decode_le(Int8, m.buffer, m.offset + 65))
end
@inline indiciesFormat!(m::SparseVectorMessageEncoder, value::Format.SbeEnum) = encode_le(Int8, m.buffer, m.offset + 65, Int8(value))

function indexing_meta_attribute(::SparseVectorMessage, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end
indexing_id(::SparseVectorMessage) = UInt16(0x4)
indexing_since_version(::SparseVectorMessage) = UInt16(0x0)
indexing_in_acting_version(m::SparseVectorMessage) = sbe_acting_version(m) >= UInt16(0x0)
indexing_encoding_offset(::SparseVectorMessage) = 66
indexing_encoding_length(::SparseVectorMessage) = 1
@inline function indexing(::Type{Integer}, m::SparseVectorMessageDecoder)
    return decode_le(Int8, m.buffer, m.offset + 66)
end
@inline function indexing(m::SparseVectorMessageDecoder)
    return Indexing.SbeEnum(decode_le(Int8, m.buffer, m.offset + 66))
end
@inline indexing!(m::SparseVectorMessageEncoder, value::Indexing.SbeEnum) = encode_le(Int8, m.buffer, m.offset + 66, Int8(value))

function reserved1_meta_attribute(::SparseVectorMessage, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end
reserved1_id(::SparseVectorMessage) = UInt16(0x5)
reserved1_since_version(::SparseVectorMessage) = UInt16(0x0)
reserved1_in_acting_version(m::SparseVectorMessage) = sbe_acting_version(m) >= UInt16(0x0)
reserved1_encoding_offset(::SparseVectorMessage) = 67
reserved1_null_value(::SparseVectorMessage) = Int8(-128)
reserved1_min_value(::SparseVectorMessage) = Int8(-127)
reserved1_max_value(::SparseVectorMessage) = Int8(127)
reserved1_encoding_length(::SparseVectorMessage) = 1

@inline function reserved1(m::SparseVectorMessageDecoder)
    return decode_le(Int8, m.buffer, m.offset + 67)
end
@inline reserved1!(m::SparseVectorMessageEncoder, value) = encode_le(Int8, m.buffer, m.offset + 67, value)

function length_meta_attribute(::SparseVectorMessage, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end
length_id(::SparseVectorMessage) = UInt16(0x6)
length_since_version(::SparseVectorMessage) = UInt16(0x0)
length_in_acting_version(m::SparseVectorMessage) = sbe_acting_version(m) >= UInt16(0x0)
length_encoding_offset(::SparseVectorMessage) = 68
length_null_value(::SparseVectorMessage) = Int64(-9223372036854775808)
length_min_value(::SparseVectorMessage) = Int64(-9223372036854775807)
length_max_value(::SparseVectorMessage) = Int64(9223372036854775807)
length_encoding_length(::SparseVectorMessage) = 8

@inline function length(m::SparseVectorMessageDecoder)
    return decode_le(Int64, m.buffer, m.offset + 68)
end
@inline length!(m::SparseVectorMessageEncoder, value) = encode_le(Int64, m.buffer, m.offset + 68, value)

function indicies_meta_attribute(::SparseVectorMessage, meta_attribute)
    meta_attribute === :semantic_type && return Symbol("int64")
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end

indicies_character_encoding(::SparseVectorMessage) = "null"
indicies_in_acting_version(m::SparseVectorMessage) = sbe_acting_version(m) >= 0
indicies_id(::SparseVectorMessage) = 20
indicies_header_length(::SparseVectorMessage) = 4

@inline function indicies_length(m::SparseVectorMessage)
    return decode_le(UInt32, m.buffer, sbe_position(m))
end

@inline function indicies_length!(m::SparseVectorMessageEncoder, n)
    if !checkbounds(Bool, m.buffer, sbe_position(m) + 1 + 4 + n)
        error("buffer too short for data length")
    elseif n > 1073741824
        error("data length too large for length type")
    end
    return encode_le(UInt32, m.buffer, sbe_position(m), n)
end

@inline function skip_indicies!(m::SparseVectorMessage)
    len = indicies_length(m)
    pos = sbe_position(m) + len + 4
    sbe_position!(m, pos)
    return len
end

@inline function indicies(m::SparseVectorMessageDecoder)
    len = indicies_length(m)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    return view(m.buffer, pos+1:pos+len)
end

@inline function indicies!(m::SparseVectorMessageEncoder, len::Int)
    indicies_length!(m, len)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    return view(m.buffer, pos+1:pos+len)
end

@inline function indicies!(m::SparseVectorMessageEncoder, src)
    len = Base.length(src)
    indicies_length!(m, len)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    dest = view(m.buffer, pos+1:pos+len)
    copyto!(dest, src)
end

function values_meta_attribute(::SparseVectorMessage, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end

values_character_encoding(::SparseVectorMessage) = "null"
values_in_acting_version(m::SparseVectorMessage) = sbe_acting_version(m) >= 0
values_id(::SparseVectorMessage) = 30
values_header_length(::SparseVectorMessage) = 4

@inline function values_length(m::SparseVectorMessage)
    return decode_le(UInt32, m.buffer, sbe_position(m))
end

@inline function values_length!(m::SparseVectorMessageEncoder, n)
    if !checkbounds(Bool, m.buffer, sbe_position(m) + 1 + 4 + n)
        error("buffer too short for data length")
    elseif n > 1073741824
        error("data length too large for length type")
    end
    return encode_le(UInt32, m.buffer, sbe_position(m), n)
end

@inline function skip_values!(m::SparseVectorMessage)
    len = values_length(m)
    pos = sbe_position(m) + len + 4
    sbe_position!(m, pos)
    return len
end

@inline function values(m::SparseVectorMessageDecoder)
    len = values_length(m)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    return view(m.buffer, pos+1:pos+len)
end

@inline function values!(m::SparseVectorMessageEncoder, len::Int)
    values_length!(m, len)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    return view(m.buffer, pos+1:pos+len)
end

@inline function values!(m::SparseVectorMessageEncoder, src)
    len = Base.length(src)
    values_length!(m, len)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    dest = view(m.buffer, pos+1:pos+len)
    copyto!(dest, src)
end

function show(io::IO, m::SparseVectorMessage{T}) where {T}
    println(io, "SparseVectorMessage view over a type $T")
    println(io, "SbeBlockLength: ", sbe_block_length(m))
    println(io, "SbeTemplateId:  ", sbe_template_id(m))
    println(io, "SbeSchemaId:    ", sbe_schema_id(m))
    println(io, "SbeSchemaVersion: ", sbe_schema_version(m))

    writer = SparseVectorMessageDecoder(sbe_buffer(m), sbe_offset(m), sbe_block_length(m), sbe_schema_version(m))
    print(io, "header: ")
    show(io, header(writer))

    println(io)
    print(io, "format: ")
    print(io, format(writer))

    println(io)
    print(io, "indiciesFormat: ")
    print(io, indiciesFormat(writer))

    println(io)
    print(io, "indexing: ")
    print(io, indexing(writer))

    println(io)
    print(io, "reserved1: ")
    print(io, reserved1(writer))

    println(io)
    print(io, "length: ")
    print(io, length(writer))

    println(io)
    print(io, "indicies: ")
    print(io, skip_indicies!(writer))
    print(io, " bytes of raw data")

    println(io)
    print(io, "values: ")
    print(io, skip_values!(writer))
    print(io, " bytes of raw data")

    nothing
end

@inline function skip!(m::SparseVectorMessage)
    skip_indicies!(m)
    skip_values!(m)
    return
end
