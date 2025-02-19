# Generated SBE (Simple Binary Encoding) message codec
# Code generated by SBE. DO NOT EDIT.

abstract type DiagonalMatrixMessage{T} end

struct DiagonalMatrixMessageDecoder{T<:AbstractArray{UInt8}} <: DiagonalMatrixMessage{T}
    buffer::T
    offset::Int64
    position_ptr::Base.RefValue{Int64}
    acting_block_length::UInt16
    acting_version::UInt16
    function DiagonalMatrixMessageDecoder(buffer::T, offset::Int64, position_ptr::Base.RefValue{Int64},
        acting_block_length::Integer, acting_version::Integer) where {T}
        position_ptr[] = offset + acting_block_length
        new{T}(buffer, offset, position_ptr, acting_block_length, acting_version)
    end
end

struct DiagonalMatrixMessageEncoder{T<:AbstractArray{UInt8}} <: DiagonalMatrixMessage{T}
    buffer::T
    offset::Int64
    position_ptr::Base.RefValue{Int64}
    function DiagonalMatrixMessageEncoder(buffer::T, offset::Int64, position_ptr::Base.RefValue{Int64}) where {T}
        position_ptr[] = offset + 68
        new{T}(buffer, offset, position_ptr)
    end
end

@inline function DiagonalMatrixMessageDecoder(buffer::AbstractArray, offset::Int64, position_ptr::Base.RefValue{Int64},
    hdr::MessageHeader)
    if templateId(hdr) != UInt16(0x4) || schemaId(hdr) != UInt16(0x1)
        error("Template id or schema id mismatch")
    end
    DiagonalMatrixMessageDecoder(buffer, offset + sbe_encoded_length(hdr), position_ptr,
        blockLength(hdr), version(hdr))
end
@inline function DiagonalMatrixMessageDecoder(buffer::AbstractArray, position_ptr::Base.RefValue{Int64},
    hdr::MessageHeader)
    DiagonalMatrixMessageDecoder(buffer, 0, position_ptr, hdr)
end
@inline function DiagonalMatrixMessageDecoder(buffer::AbstractArray, offset::Int64,
    acting_block_length::Integer, acting_version::Integer)
    DiagonalMatrixMessageDecoder(buffer, offset, Ref(0), acting_block_length, acting_version)
end
@inline function DiagonalMatrixMessageDecoder(buffer::AbstractArray, offset::Int64, hdr::MessageHeader)
    DiagonalMatrixMessageDecoder(buffer, offset, Ref(0), hdr)
end
@inline DiagonalMatrixMessageDecoder(buffer::AbstractArray, hdr::MessageHeader) = DiagonalMatrixMessageDecoder(buffer, 0, Ref(0), hdr)
@inline function DiagonalMatrixMessageEncoder(buffer::AbstractArray, position_ptr::Base.RefValue{Int64})
    DiagonalMatrixMessageEncoder(buffer, 0, position_ptr)
end
@inline function DiagonalMatrixMessageEncoder(buffer::AbstractArray, offset::Int64, position_ptr::Base.RefValue{Int64},
    hdr::MessageHeader)
    blockLength!(hdr, UInt16(0x44))
    templateId!(hdr, UInt16(0x4))
    schemaId!(hdr, UInt16(0x1))
    version!(hdr, UInt16(0x0))
    DiagonalMatrixMessageEncoder(buffer, offset + sbe_encoded_length(hdr), position_ptr)
end
@inline function DiagonalMatrixMessageEncoder(buffer::AbstractArray, position_ptr::Base.RefValue{Int64},    hdr::MessageHeader)
    DiagonalMatrixMessageEncoder(buffer, 0, position_ptr, hdr)
end
@inline function DiagonalMatrixMessageEncoder(buffer::AbstractArray, offset::Int64, hdr::MessageHeader)
    DiagonalMatrixMessageEncoder(buffer, offset, Ref(0), hdr)
end
@inline function DiagonalMatrixMessageEncoder(buffer::AbstractArray, hdr::MessageHeader)
    DiagonalMatrixMessageEncoder(buffer, 0, Ref(0), hdr)
end
@inline DiagonalMatrixMessageEncoder(buffer::AbstractArray, offset::Int64=0) = DiagonalMatrixMessageEncoder(buffer, offset, Ref(0))
sbe_buffer(m::DiagonalMatrixMessage) = m.buffer
sbe_offset(m::DiagonalMatrixMessage) = m.offset
sbe_position_ptr(m::DiagonalMatrixMessage) = m.position_ptr
sbe_position(m::DiagonalMatrixMessage) = m.position_ptr[]
sbe_position!(m::DiagonalMatrixMessage, position) = m.position_ptr[] = position
sbe_block_length(::DiagonalMatrixMessage) = UInt16(0x44)
sbe_block_length(::Type{<:DiagonalMatrixMessage}) = UInt16(0x44)
sbe_template_id(::DiagonalMatrixMessage) = UInt16(0x4)
sbe_template_id(::Type{<:DiagonalMatrixMessage})  = UInt16(0x4)
sbe_schema_id(::DiagonalMatrixMessage) = UInt16(0x1)
sbe_schema_id(::Type{<:DiagonalMatrixMessage})  = UInt16(0x1)
sbe_schema_version(::DiagonalMatrixMessage) = UInt16(0x0)
sbe_schema_version(::Type{<:DiagonalMatrixMessage})  = UInt16(0x0)
sbe_semantic_type(::DiagonalMatrixMessage) = ""
sbe_semantic_version(::DiagonalMatrixMessage) = ""
sbe_acting_block_length(m::DiagonalMatrixMessageDecoder) = m.acting_block_length
sbe_acting_block_length(::DiagonalMatrixMessageEncoder) = UInt16(0x44)
sbe_acting_version(m::DiagonalMatrixMessageDecoder) = m.acting_version
sbe_acting_version(::DiagonalMatrixMessageEncoder) = UInt16(0x0)
sbe_rewind!(m::DiagonalMatrixMessage) = sbe_position!(m, m.offset + sbe_acting_block_length(m))
sbe_encoded_length(m::DiagonalMatrixMessage) = sbe_position(m) - m.offset
@inline function sbe_decoded_length(m::DiagonalMatrixMessage)
    skipper = DiagonalMatrixMessageDecoder(sbe_buffer(m), sbe_offset(m),
        sbe_acting_block_length(m), sbe_acting_version(m))
    sbe_skip!(skipper)
    sbe_encoded_length(skipper)
end

function header_meta_attribute(::DiagonalMatrixMessage, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end
header_id(::DiagonalMatrixMessage) = UInt16(0x1)
header_since_version(::DiagonalMatrixMessage) = UInt16(0x0)
header_in_acting_version(m::DiagonalMatrixMessage) = sbe_acting_version(m) >= UInt16(0x0)
header_encoding_offset(::DiagonalMatrixMessage) = 0
header(m::DiagonalMatrixMessage) = SpidersMessageHeader(m.buffer, m.offset + 0, sbe_acting_version(m))

function format_meta_attribute(::DiagonalMatrixMessage, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end
format_id(::DiagonalMatrixMessage) = UInt16(0x2)
format_since_version(::DiagonalMatrixMessage) = UInt16(0x0)
format_in_acting_version(m::DiagonalMatrixMessage) = sbe_acting_version(m) >= UInt16(0x0)
format_encoding_offset(::DiagonalMatrixMessage) = 64
format_encoding_length(::DiagonalMatrixMessage) = 1
@inline function format(::Type{Integer}, m::DiagonalMatrixMessageDecoder)
    return decode_le(Int8, m.buffer, m.offset + 64)
end
@inline function format(m::DiagonalMatrixMessageDecoder)
    return Format.SbeEnum(decode_le(Int8, m.buffer, m.offset + 64))
end
@inline format!(m::DiagonalMatrixMessageEncoder, value::Format.SbeEnum) = encode_le(Int8, m.buffer, m.offset + 64, Int8(value))

function indiciesFormat_meta_attribute(::DiagonalMatrixMessage, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end
indiciesFormat_id(::DiagonalMatrixMessage) = UInt16(0x3)
indiciesFormat_since_version(::DiagonalMatrixMessage) = UInt16(0x0)
indiciesFormat_in_acting_version(m::DiagonalMatrixMessage) = sbe_acting_version(m) >= UInt16(0x0)
indiciesFormat_encoding_offset(::DiagonalMatrixMessage) = 65
indiciesFormat_encoding_length(::DiagonalMatrixMessage) = 1
@inline function indiciesFormat(::Type{Integer}, m::DiagonalMatrixMessageDecoder)
    return decode_le(Int8, m.buffer, m.offset + 65)
end
@inline function indiciesFormat(m::DiagonalMatrixMessageDecoder)
    return Format.SbeEnum(decode_le(Int8, m.buffer, m.offset + 65))
end
@inline indiciesFormat!(m::DiagonalMatrixMessageEncoder, value::Format.SbeEnum) = encode_le(Int8, m.buffer, m.offset + 65, Int8(value))

function reserved1_meta_attribute(::DiagonalMatrixMessage, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end
reserved1_id(::DiagonalMatrixMessage) = UInt16(0xa)
reserved1_since_version(::DiagonalMatrixMessage) = UInt16(0x0)
reserved1_in_acting_version(m::DiagonalMatrixMessage) = sbe_acting_version(m) >= UInt16(0x0)
reserved1_encoding_offset(::DiagonalMatrixMessage) = 66
reserved1_null_value(::DiagonalMatrixMessage) = Int8(-128)
reserved1_min_value(::DiagonalMatrixMessage) = Int8(-127)
reserved1_max_value(::DiagonalMatrixMessage) = Int8(127)
reserved1_encoding_length(::DiagonalMatrixMessage) = 2
reserved1_length(::DiagonalMatrixMessage) = 2
reserved1_eltype(::DiagonalMatrixMessage) = Int8

@inline function reserved1(m::DiagonalMatrixMessageDecoder)
    return mappedarray(ltoh, reinterpret(Int8, view(m.buffer, m.offset+66+1:m.offset+66+sizeof(Int8)*2)))
end

@inline function reserved1(::Type{<:SVector},m::DiagonalMatrixMessageDecoder)
    return mappedarray(ltoh, reinterpret(SVector{2,Int8}, view(m.buffer, m.offset+66+1:m.offset+66+sizeof(Int8)*2))[])
end

@inline function reserved1!(m::DiagonalMatrixMessageEncoder)
    return mappedarray(ltoh, htol, reinterpret(Int8, view(m.buffer, m.offset+66+1:m.offset+66+sizeof(Int8)*2)))
end

@inline function reserved1!(m::DiagonalMatrixMessageEncoder, value)
    copyto!(mappedarray(ltoh, htol, reinterpret(Int8, view(m.buffer, m.offset+66+1:m.offset+66+sizeof(Int8)*2))), value)
end

function dims_meta_attribute(::DiagonalMatrixMessage, meta_attribute)
    meta_attribute === :semantic_type && return Symbol("int64")
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end

dims_character_encoding(::DiagonalMatrixMessage) = "null"
dims_in_acting_version(m::DiagonalMatrixMessage) = sbe_acting_version(m) >= 0
dims_id(::DiagonalMatrixMessage) = 20
dims_header_length(::DiagonalMatrixMessage) = 4

@inline function dims_length(m::DiagonalMatrixMessage)
    return decode_le(UInt32, m.buffer, sbe_position(m))
end

@inline function dims_length!(m::DiagonalMatrixMessageEncoder, n)
    if !checkbounds(Bool, m.buffer, sbe_position(m) + 4 + n)
        error("buffer too short for data length")
    elseif n > 1073741824
        error("data length too large for length type")
    end
    return encode_le(UInt32, m.buffer, sbe_position(m), n)
end

@inline function skip_dims!(m::DiagonalMatrixMessage)
    len = dims_length(m)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    return len
end

@inline function dims(m::DiagonalMatrixMessageDecoder)
    len = dims_length(m)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    return view(m.buffer, pos+1:pos+len)
end

@inline function dims!(m::DiagonalMatrixMessageEncoder, len::Int64)
    dims_length!(m, len)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    return view(m.buffer, pos+1:pos+len)
end

@inline function dims!(m::DiagonalMatrixMessageEncoder, src)
    len = Base.length(src)
    dims_length!(m, len)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    dest = view(m.buffer, pos+1:pos+len)
    copyto!(dest, src)
end

function values_meta_attribute(::DiagonalMatrixMessage, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end

values_character_encoding(::DiagonalMatrixMessage) = "null"
values_in_acting_version(m::DiagonalMatrixMessage) = sbe_acting_version(m) >= 0
values_id(::DiagonalMatrixMessage) = 30
values_header_length(::DiagonalMatrixMessage) = 4

@inline function values_length(m::DiagonalMatrixMessage)
    return decode_le(UInt32, m.buffer, sbe_position(m))
end

@inline function values_length!(m::DiagonalMatrixMessageEncoder, n)
    if !checkbounds(Bool, m.buffer, sbe_position(m) + 4 + n)
        error("buffer too short for data length")
    elseif n > 1073741824
        error("data length too large for length type")
    end
    return encode_le(UInt32, m.buffer, sbe_position(m), n)
end

@inline function skip_values!(m::DiagonalMatrixMessage)
    len = values_length(m)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    return len
end

@inline function values(m::DiagonalMatrixMessageDecoder)
    len = values_length(m)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    return view(m.buffer, pos+1:pos+len)
end

@inline function values!(m::DiagonalMatrixMessageEncoder, len::Int64)
    values_length!(m, len)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    return view(m.buffer, pos+1:pos+len)
end

@inline function values!(m::DiagonalMatrixMessageEncoder, src)
    len = Base.length(src)
    values_length!(m, len)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    dest = view(m.buffer, pos+1:pos+len)
    copyto!(dest, src)
end

function show(io::IO, m::DiagonalMatrixMessage{T}) where {T}
    println(io, "DiagonalMatrixMessage view over a type $T")
    println(io, "SbeBlockLength: ", sbe_block_length(m))
    println(io, "SbeTemplateId:  ", sbe_template_id(m))
    println(io, "SbeSchemaId:    ", sbe_schema_id(m))
    println(io, "SbeSchemaVersion: ", sbe_schema_version(m))

    writer = DiagonalMatrixMessageDecoder(sbe_buffer(m), sbe_offset(m), sbe_block_length(m), sbe_schema_version(m))
    print(io, "header: ")
    show(io, header(writer))

    println(io)
    print(io, "format: ")
    print(io, format(writer))

    println(io)
    print(io, "indiciesFormat: ")
    print(io, indiciesFormat(writer))

    println(io)
    print(io, "reserved1: ")
    print(io, reserved1(writer))

    println(io)
    print(io, "dims: ")
    print(io, skip_dims!(writer))
    print(io, " bytes of raw data")

    println(io)
    print(io, "values: ")
    print(io, skip_values!(writer))
    print(io, " bytes of raw data")

    nothing
end

@inline function sbe_skip!(m::DiagonalMatrixMessage)
    sbe_rewind!(m)
    skip_dims!(m)
    skip_values!(m)
    return
end
