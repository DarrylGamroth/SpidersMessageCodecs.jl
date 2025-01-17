# Generated SBE (Simple Binary Encoding) message codec
# Code generated by SBE. DO NOT EDIT.

abstract type TensorMessage{T} end

struct TensorMessageDecoder{T<:AbstractArray{UInt8}} <: TensorMessage{T}
    buffer::T
    offset::Int64
    position_ptr::Base.RefValue{Int64}
    acting_block_length::UInt16
    acting_version::UInt16
    function TensorMessageDecoder(buffer::T, offset::Int64, position_ptr::Base.RefValue{Int64},
        acting_block_length::Integer, acting_version::Integer) where {T}
        position_ptr[] = offset + acting_block_length
        new{T}(buffer, offset, position_ptr, acting_block_length, acting_version)
    end
end

struct TensorMessageEncoder{T<:AbstractArray{UInt8}} <: TensorMessage{T}
    buffer::T
    offset::Int64
    position_ptr::Base.RefValue{Int64}
    function TensorMessageEncoder(buffer::T, offset::Int64, position_ptr::Base.RefValue{Int64}) where {T}
        position_ptr[] = offset + 68
        new{T}(buffer, offset, position_ptr)
    end
end

function TensorMessageDecoder(buffer::AbstractArray, offset::Int64, position_ptr::Base.RefValue{Int64},
    hdr::MessageHeader)
    if templateId(hdr) != UInt16(0x1) || schemaId(hdr) != UInt16(0x1)
        error("Template id or schema id mismatch")
    end
    TensorMessageDecoder(buffer, offset + sbe_encoded_length(hdr), position_ptr,
        blockLength(hdr), version(hdr))
end
function TensorMessageDecoder(buffer::AbstractArray, position_ptr::Base.RefValue{Int64},
    hdr::MessageHeader)
    TensorMessageDecoder(buffer, 0, position_ptr, hdr)
end
function TensorMessageDecoder(buffer::AbstractArray, offset::Int64,
    acting_block_length::Integer, acting_version::Integer)
    TensorMessageDecoder(buffer, offset, Ref(0), acting_block_length, acting_version)
end
function TensorMessageDecoder(buffer::AbstractArray, offset::Int64, hdr::MessageHeader)
    TensorMessageDecoder(buffer, offset, Ref(0), hdr)
end
TensorMessageDecoder(buffer::AbstractArray, hdr::MessageHeader) = TensorMessageDecoder(buffer, 0, Ref(0), hdr)
function TensorMessageEncoder(buffer::AbstractArray, position_ptr::Base.RefValue{Int64})
    TensorMessageEncoder(buffer, 0, position_ptr)
end
function TensorMessageEncoder(buffer::AbstractArray, offset::Int64, position_ptr::Base.RefValue{Int64},
    hdr::MessageHeader)
    blockLength!(hdr, UInt16(0x44))
    templateId!(hdr, UInt16(0x1))
    schemaId!(hdr, UInt16(0x1))
    version!(hdr, UInt16(0x0))
    TensorMessageEncoder(buffer, offset + sbe_encoded_length(hdr), position_ptr)
end
function TensorMessageEncoder(buffer::AbstractArray, position_ptr::Base.RefValue{Int64}, hdr::MessageHeader)
    TensorMessageEncoder(buffer, 0, position_ptr, hdr)
end
function TensorMessageEncoder(buffer::AbstractArray, offset::Int64, hdr::MessageHeader)
    TensorMessageEncoder(buffer, offset, Ref(0), hdr)
end
function TensorMessageEncoder(buffer::AbstractArray, hdr::MessageHeader)
    TensorMessageEncoder(buffer, 0, Ref(0), hdr)
end
TensorMessageEncoder(buffer::AbstractArray, offset::Int64=0) = TensorMessageEncoder(buffer, offset, Ref(0))
sbe_buffer(m::TensorMessage) = m.buffer
sbe_offset(m::TensorMessage) = m.offset
sbe_position_ptr(m::TensorMessage) = m.position_ptr
sbe_position(m::TensorMessage) = m.position_ptr[]
sbe_position!(m::TensorMessage, position) = m.position_ptr[] = position
sbe_block_length(::TensorMessage) = UInt16(0x44)
sbe_block_length(::Type{<:TensorMessage}) = UInt16(0x44)
sbe_template_id(::TensorMessage) = UInt16(0x1)
sbe_template_id(::Type{<:TensorMessage})  = UInt16(0x1)
sbe_schema_id(::TensorMessage) = UInt16(0x1)
sbe_schema_id(::Type{<:TensorMessage})  = UInt16(0x1)
sbe_schema_version(::TensorMessage) = UInt16(0x0)
sbe_schema_version(::Type{<:TensorMessage})  = UInt16(0x0)
sbe_semantic_type(::TensorMessage) = ""
sbe_semantic_version(::TensorMessage) = ""
sbe_acting_block_length(m::TensorMessageDecoder) = m.acting_block_length
sbe_acting_block_length(::TensorMessageEncoder) = UInt16(0x44)
sbe_acting_version(m::TensorMessageDecoder) = m.acting_version
sbe_acting_version(::TensorMessageEncoder) = UInt16(0x0)
sbe_rewind!(m::TensorMessage) = sbe_position!(m, m.offset + sbe_acting_block_length(m))
sbe_encoded_length(m::TensorMessage) = sbe_position(m) - m.offset
@inline function sbe_decoded_length(m::TensorMessage)
    skipper = TensorMessageDecoder(sbe_buffer(m), sbe_offset(m),
        sbe_acting_block_length(m), sbe_acting_version(m))
    sbe_rewind!(skipper)
    skip!(skipper)
    sbe_encoded_length(skipper)
end

function header_meta_attribute(::TensorMessage, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end
header_id(::TensorMessage) = UInt16(0x1)
header_since_version(::TensorMessage) = UInt16(0x0)
header_in_acting_version(m::TensorMessage) = sbe_acting_version(m) >= UInt16(0x0)
header_encoding_offset(::TensorMessage) = 0
header(m::TensorMessage) = SpidersMessageHeader(m.buffer, m.offset + 0, sbe_acting_version(m))

function format_meta_attribute(::TensorMessage, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end
format_id(::TensorMessage) = UInt16(0x2)
format_since_version(::TensorMessage) = UInt16(0x0)
format_in_acting_version(m::TensorMessage) = sbe_acting_version(m) >= UInt16(0x0)
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
    return Symbol("")
end
order_id(::TensorMessage) = UInt16(0x3)
order_since_version(::TensorMessage) = UInt16(0x0)
order_in_acting_version(m::TensorMessage) = sbe_acting_version(m) >= UInt16(0x0)
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
    return Symbol("")
end
reserved1_id(::TensorMessage) = UInt16(0x4)
reserved1_since_version(::TensorMessage) = UInt16(0x0)
reserved1_in_acting_version(m::TensorMessage) = sbe_acting_version(m) >= UInt16(0x0)
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

function dims_meta_attribute(::TensorMessage, meta_attribute)
    meta_attribute === :semantic_type && return Symbol("int32")
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end

dims_character_encoding(::TensorMessage) = "null"
dims_in_acting_version(m::TensorMessage) = sbe_acting_version(m) >= 0
dims_id(::TensorMessage) = 20
dims_header_length(::TensorMessage) = 4

@inline function dims_length(m::TensorMessage)
    return decode_le(UInt32, m.buffer, sbe_position(m))
end

@inline function dims_length!(m::TensorMessageEncoder, n)
    if !checkbounds(Bool, m.buffer, sbe_position(m) + 1 + 4 + n)
        error("buffer too short for data length")
    elseif n > 1073741824
        error("data length too large for length type")
    end
    return encode_le(UInt32, m.buffer, sbe_position(m), n)
end

@inline function skip_dims!(m::TensorMessage)
    len = dims_length(m)
    pos = sbe_position(m) + len + 4
    sbe_position!(m, pos)
    return len
end

@inline function dims(m::TensorMessageDecoder)
    len = dims_length(m)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    return view(m.buffer, pos+1:pos+len)
end

@inline function dims!(m::TensorMessageEncoder, len::Int64)
    dims_length!(m, len)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    return view(m.buffer, pos+1:pos+len)
end

@inline function dims!(m::TensorMessageEncoder, src)
    len = Base.length(src)
    dims_length!(m, len)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    dest = view(m.buffer, pos+1:pos+len)
    copyto!(dest, src)
end

function offset_meta_attribute(::TensorMessage, meta_attribute)
    meta_attribute === :semantic_type && return Symbol("int32")
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
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
    elseif n > 1073741824
        error("data length too large for length type")
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

@inline function offset!(m::TensorMessageEncoder, len::Int64)
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
    return Symbol("")
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
    elseif n > 1073741824
        error("data length too large for length type")
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

@inline function values!(m::TensorMessageEncoder, len::Int64)
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

function show(io::IO, m::TensorMessage{T}) where {T}
    println(io, "TensorMessage view over a type $T")
    println(io, "SbeBlockLength: ", sbe_block_length(m))
    println(io, "SbeTemplateId:  ", sbe_template_id(m))
    println(io, "SbeSchemaId:    ", sbe_schema_id(m))
    println(io, "SbeSchemaVersion: ", sbe_schema_version(m))

    writer = TensorMessageDecoder(sbe_buffer(m), sbe_offset(m), sbe_block_length(m), sbe_schema_version(m))
    print(io, "header: ")
    show(io, header(writer))

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
    print(io, "dims: ")
    print(io, skip_dims!(writer))
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
    skip_dims!(m)
    skip_offset!(m)
    skip_values!(m)
    return
end
