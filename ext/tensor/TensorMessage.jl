# Generated SBE (Simple Binary Encoding) message codec
# Code generated by SBE. DO NOT EDIT.

abstract type TensorMessage{T} end

struct TensorMessageDecoder{T<:AbstractArray{UInt8}} <: TensorMessage{T}
    buffer::T
    offset::Int64
    position_ptr::Base.RefValue{Int64}
    acting_block_length::Int64
    acting_version::Int64
end

struct TensorMessageEncoder{T<:AbstractArray{UInt8}} <: TensorMessage{T}
    buffer::T
    offset::Int64
    position_ptr::Base.RefValue{Int64}
end

@inline function TensorMessageDecoder(context::SbeCodecContext, buffer, offset,
    acting_block_length, acting_version)
    sbe_position!(context, offset + acting_block_length)
    TensorMessageDecoder(buffer, offset, sbe_position_ptr(context), acting_block_length, acting_version)
end

@inline function TensorMessageDecoder(context::SbeCodecContext, buffer, offset, hdr::MessageHeader)
    TensorMessageDecoder(context, buffer, offset + sbe_encoded_length(hdr),
        Int64(blockLength(hdr)), Int64(version(hdr)))
end

@inline function TensorMessageDecoder(context::SbeCodecContext, buffer, hdr::MessageHeader)
    TensorMessageDecoder(context, buffer, 0, hdr)
end
@inline function TensorMessageDecoder(buffer, offset, acting_block_length, acting_version)
    TensorMessageDecoder(SbeCodecContext(), buffer, offset, acting_block_length, acting_version)
end

@inline TensorMessageDecoder(buffer, offset, hdr) = TensorMessageDecoder(SbeCodecContext(), buffer, offset, hdr)
@inline TensorMessageDecoder(buffer, hdr) = TensorMessageDecoder(SbeCodecContext(), buffer, 0, hdr)
 @inline function TensorMessageEncoder(context::SbeCodecContext, buffer, offset=0)
    sbe_position!(context, offset + 68)
    TensorMessageEncoder(buffer, offset, sbe_position_ptr(context))
end
@inline TensorMessageEncoder(buffer, offset=0) = TensorMessageEncoder(SbeCodecContext(), buffer, offset)
@inline TensorMessage() = TensorMessageEncoder(SbeCodecContext(), UInt8[])

@inline function TensorMessageEncoder(context::SbeCodecContext, buffer, offset, hdr::MessageHeader)
    blockLength!(hdr, 68)
    templateId!(hdr, 1)
    schemaId!(hdr, 1)
    version!(hdr, 0)
    TensorMessageEncoder(context, buffer, offset + sbe_encoded_length(hdr))
end

@inline function TensorMessageEncoder(context::SbeCodecContext, buffer, hdr::MessageHeader)
    TensorMessageEncoder(context, buffer, 0, hdr)
end

@inline function TensorMessageEncoder(buffer, offset, hdr::MessageHeader)
    TensorMessageEncoder(SbeCodecContext(), buffer, offset, hdr)
end

@inline function TensorMessageEncoder(buffer, hdr::MessageHeader)
    TensorMessageEncoder(SbeCodecContext(), buffer, 0, hdr)
end

sbe_buffer(m::TensorMessage) = m.buffer
sbe_offset(m::TensorMessage) = m.offset
sbe_position_ptr(m::TensorMessage) = m.position_ptr
sbe_position(m::TensorMessage) = m.position_ptr[]
@inline sbe_check_position(m::TensorMessage, position) = (checkbounds(m.buffer, position + 1); position)
@inline sbe_position!(m::TensorMessage, position) = m.position_ptr[] = position
sbe_block_length(::TensorMessage) = 68
sbe_template_id(::TensorMessage) = 1
sbe_schema_id(::TensorMessage) = 1
sbe_schema_version(::TensorMessage) = 0
sbe_semantic_type(::TensorMessage) = ""
sbe_semantic_version(::TensorMessage) = ""
sbe_encoded_length(m::TensorMessageEncoder) = sbe_position(m) - m.offset
sbe_rewind!(m::TensorMessageEncoder) = sbe_position!(m, m.offset + 68)

sbe_acting_block_length(m::TensorMessageDecoder) = m.acting_block_length
sbe_acting_version(m::TensorMessageDecoder) = m.acting_version
sbe_rewind!(m::TensorMessageDecoder) = sbe_position!(m, m.offset + m.acting_block_length)

@inline function sbe_decoded_length(m::TensorMessageDecoder)
    skipper = TensorMessageEncoder(m.buffer, m.offset)
    skip!(skipper)
    return sbe_encoded_length(skipper)
end

function sbe_decoded_buffer(m::TensorMessageDecoder)
    offset = m.offset - sbe_encoded_length(MessageHeader())
    offset < 0 && throw(ArgumentError("Message offset is negative"))
    return view(m.buffer, offset+1:offset+sbe_decoded_length(m))
end

function sbe_encoded_buffer(m::TensorMessageEncoder)
    offset = m.offset - sbe_encoded_length(MessageHeader())
    offset < 0 && throw(ArgumentError("Message offset is negative"))
    d = TensorMessageDecoder(m.buffer, MessageHeader(m.buffer, offset))
    return view(d.buffer, offset+1:offset+sbe_decoded_length(d))
end

function header_meta_attribute(::TensorMessage, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    error(lazy"unknown attribute: $meta_attribute")
end
header_id(::TensorMessage) = 1
header_since_version(::TensorMessage) = 0
header_in_acting_version(m::TensorMessage) = sbe_acting_version(m) >= 0
header_encoding_offset(::TensorMessage) = 0
header(m::TensorMessage) = SpidersMessageHeader(m.buffer, m.offset + 0)

function format_meta_attribute(::TensorMessage, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    error(lazy"unknown attribute: $meta_attribute")
end
format_id(::TensorMessage) = 2
format_since_version(::TensorMessage) = 0
format_in_acting_version(m::TensorMessage) = sbe_acting_version(m) >= 0
format_encoding_offset(::TensorMessage) = 64
format_encoding_length(::TensorMessage) = 1
@inline function format(::Type{Integer}, m::TensorMessageDecoder)
    return decode_le(Int8, m.buffer, m.offset + 64)
end
@inline function format(m::TensorMessageDecoder)
    return Format.SbeEnum(decode_le(Int8, m.buffer, m.offset + 64))
end
@inline format!(m::TensorMessageEncoder, value::Format.SbeEnum) = encode_le(Int8, m.buffer, m.offset + 64, Int8(value))

function order_meta_attribute(::TensorMessage, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    error(lazy"unknown attribute: $meta_attribute")
end
order_id(::TensorMessage) = 3
order_since_version(::TensorMessage) = 0
order_in_acting_version(m::TensorMessage) = sbe_acting_version(m) >= 0
order_encoding_offset(::TensorMessage) = 65
order_encoding_length(::TensorMessage) = 1
@inline function order(::Type{Integer}, m::TensorMessageDecoder)
    return decode_le(Int8, m.buffer, m.offset + 65)
end
@inline function order(m::TensorMessageDecoder)
    return Order.SbeEnum(decode_le(Int8, m.buffer, m.offset + 65))
end
@inline order!(m::TensorMessageEncoder, value::Order.SbeEnum) = encode_le(Int8, m.buffer, m.offset + 65, Int8(value))

function reserved1_meta_attribute(::TensorMessage, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    error(lazy"unknown attribute: $meta_attribute")
end
reserved1_id(::TensorMessage) = 4
reserved1_since_version(::TensorMessage) = 0
reserved1_in_acting_version(m::TensorMessage) = sbe_acting_version(m) >= 0
reserved1_encoding_offset(::TensorMessage) = 66
reserved1_null_value(::TensorMessage) = Int8(-128)
reserved1_min_value(::TensorMessage) = Int8(-127)
reserved1_max_value(::TensorMessage) = Int8(127)
reserved1_encoding_length(::TensorMessage) = 2
reserved1_length(::TensorMessage) = 2
reserved1_eltype(::TensorMessage) = Int8

@inline function reserved1(m::TensorMessageDecoder)
    return mappedarray(ltoh, reinterpret(Int8, view(m.buffer, m.offset+66+1:m.offset+66+sizeof(Int8)*2)))
end

@inline function reserved1(::Type{<:SVector},m::TensorMessageDecoder)
    return mappedarray(ltoh, reinterpret(SVector{2,Int8}, view(m.buffer, m.offset+66+1:m.offset+66+sizeof(Int8)*2))[])
end

@inline function reserved1!(m::TensorMessageEncoder)
    return mappedarray(ltoh, htol, reinterpret(Int8, view(m.buffer, m.offset+66+1:m.offset+66+sizeof(Int8)*2)))
end

@inline function reserved1!(m::TensorMessageEncoder, value)
    copyto!(mappedarray(ltoh, htol, reinterpret(Int8, view(m.buffer, m.offset+66+1:m.offset+66+sizeof(Int8)*2))), value)
end

function shape_meta_attribute(::TensorMessage, meta_attribute)
    meta_attribute === :semantic_type && return Symbol("int32")
    meta_attribute === :presence && return Symbol("required")
    error(lazy"unknown attribute: $meta_attribute")
end

shape_character_encoding(::TensorMessage) = "null"
shape_in_acting_version(m::TensorMessage) = sbe_acting_version(m) >= 0
shape_id(::TensorMessage) = 20
shape_header_length(::TensorMessage) = 4

@inline function shape_length(m::TensorMessage)
    return decode_le(UInt32, m.buffer, sbe_position(m))
end

@inline function shape_length!(m::TensorMessageEncoder, n)
    if !checkbounds(Bool, m.buffer, sbe_position(m) + 1 + 4 + n)
        error("buffer too short for data length")
    end
    return encode_le(UInt32, m.buffer, sbe_position(m), n)
end

@inline function skip_shape!(m::TensorMessage)
    len = shape_length(m)
    pos = sbe_position(m) + len + 4
    sbe_position!(m, pos)
    return len
end

@inline function shape(m::TensorMessageDecoder)
    len = shape_length(m)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    return view(m.buffer, pos+1:pos+len)
end

@inline function shape!(m::TensorMessageEncoder, len::Int)
    shape_length!(m, len)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    return view(m.buffer, pos+1:pos+len)
end

@inline function shape!(m::TensorMessageEncoder, src)
    len = Base.length(src)
    shape_length!(m, len)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    dest = view(m.buffer, pos+1:pos+len)
    copyto!(dest, src)
end

function offset_meta_attribute(::TensorMessage, meta_attribute)
    meta_attribute === :semantic_type && return Symbol("int32")
    meta_attribute === :presence && return Symbol("required")
    error(lazy"unknown attribute: $meta_attribute")
end

offset_character_encoding(::TensorMessage) = "null"
offset_in_acting_version(m::TensorMessage) = sbe_acting_version(m) >= 0
offset_id(::TensorMessage) = 21
offset_header_length(::TensorMessage) = 4

@inline function offset_length(m::TensorMessage)
    return decode_le(UInt32, m.buffer, sbe_position(m))
end

@inline function offset_length!(m::TensorMessageEncoder, n)
    if !checkbounds(Bool, m.buffer, sbe_position(m) + 1 + 4 + n)
        error("buffer too short for data length")
    end
    return encode_le(UInt32, m.buffer, sbe_position(m), n)
end

@inline function skip_offset!(m::TensorMessage)
    len = offset_length(m)
    pos = sbe_position(m) + len + 4
    sbe_position!(m, pos)
    return len
end

@inline function offset(m::TensorMessageDecoder)
    len = offset_length(m)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    return view(m.buffer, pos+1:pos+len)
end

@inline function offset!(m::TensorMessageEncoder, len::Int)
    offset_length!(m, len)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    return view(m.buffer, pos+1:pos+len)
end

@inline function offset!(m::TensorMessageEncoder, src)
    len = Base.length(src)
    offset_length!(m, len)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    dest = view(m.buffer, pos+1:pos+len)
    copyto!(dest, src)
end

function values_meta_attribute(::TensorMessage, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    error(lazy"unknown attribute: $meta_attribute")
end

values_character_encoding(::TensorMessage) = "null"
values_in_acting_version(m::TensorMessage) = sbe_acting_version(m) >= 0
values_id(::TensorMessage) = 30
values_header_length(::TensorMessage) = 4

@inline function values_length(m::TensorMessage)
    return decode_le(UInt32, m.buffer, sbe_position(m))
end

@inline function values_length!(m::TensorMessageEncoder, n)
    if !checkbounds(Bool, m.buffer, sbe_position(m) + 1 + 4 + n)
        error("buffer too short for data length")
    end
    return encode_le(UInt32, m.buffer, sbe_position(m), n)
end

@inline function skip_values!(m::TensorMessage)
    len = values_length(m)
    pos = sbe_position(m) + len + 4
    sbe_position!(m, pos)
    return len
end

@inline function values(m::TensorMessageDecoder)
    len = values_length(m)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    return view(m.buffer, pos+1:pos+len)
end

@inline function values!(m::TensorMessageEncoder, len::Int)
    values_length!(m, len)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    return view(m.buffer, pos+1:pos+len)
end

@inline function values!(m::TensorMessageEncoder, src)
    len = Base.length(src)
    values_length!(m, len)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    dest = view(m.buffer, pos+1:pos+len)
    copyto!(dest, src)
end

function Base.show(io::IO, m::TensorMessage{T}) where {T}
    println(io, "TensorMessage view over a type $T")
    println(io, "SbeBlockLength: ", sbe_block_length(m))
    println(io, "SbeTemplateId:  ", sbe_template_id(m))
    println(io, "SbeSchemaId:    ", sbe_schema_id(m))
    println(io, "SbeSchemaVersion: ", sbe_schema_version(m))

    writer = TensorMessageDecoder(sbe_buffer(m), sbe_offset(m), sbe_block_length(m), sbe_schema_version(m))
    print(io, "header: ")
    Base.show(io, header(writer))

    println(io)
    print(io, "format: ")
    print(io, format(writer))

    println(io)
    print(io, "order: ")
    print(io, order(writer))

    println(io)
    print(io, "reserved1: ")
    print(io, reserved1(writer))

    println(io)
    print(io, "shape: ")
    print(io, skip_shape!(writer))
    print(io, " bytes of raw data")

    println(io)
    print(io, "offset: ")
    print(io, skip_offset!(writer))
    print(io, " bytes of raw data")

    println(io)
    print(io, "values: ")
    print(io, skip_values!(writer))
    print(io, " bytes of raw data")

    nothing
end

@inline function skip!(m::TensorMessage)
    skip_shape!(m)
    skip_offset!(m)
    skip_values!(m)
    return
end
