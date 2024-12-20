# Generated SBE (Simple Binary Encoding) message codec
# Code generated by SBE. DO NOT EDIT.

abstract type GenericMessage{T} end

struct GenericMessageDecoder{T<:AbstractArray{UInt8}} <: GenericMessage{T}
    buffer::T
    offset::Int64
    position_ptr::Base.RefValue{Int64}
    acting_block_length::UInt16
    acting_version::UInt16
    function GenericMessageDecoder(buffer::T, offset::Int, position_ptr::Base.RefValue{Int64},
        acting_block_length::UInt16, acting_version::UInt16) where {T}
        position_ptr[] = offset + acting_block_length
        new{T}(buffer, offset, position_ptr, acting_block_length, acting_version)
    end
end

struct GenericMessageEncoder{T<:AbstractArray{UInt8}} <: GenericMessage{T}
    buffer::T
    offset::Int64
    position_ptr::Base.RefValue{Int64}
    function GenericMessageEncoder(buffer::T, offset::Int, position_ptr::Base.RefValue{Int64}) where {T}
        position_ptr[] = offset + 64
        new{T}(buffer, offset, position_ptr)
    end
end

function GenericMessageDecoder(buffer::AbstractArray, offset::Int, position_ptr::Base.RefValue{Int64},
    hdr::MessageHeader)
    if templateId(hdr) != UInt16(0x7) || schemaId(hdr) != UInt16(0x9)
        error("Template id or schema id mismatch")
    end
    GenericMessageDecoder(buffer, offset + sbe_encoded_length(hdr), position_ptr,
        blockLength(hdr), version(hdr))
end
function GenericMessageDecoder(buffer::AbstractArray, position_ptr::Base.RefValue{Int64},
    hdr::MessageHeader)
    GenericMessageDecoder(buffer, 0, position_ptr, hdr)
end
function GenericMessageDecoder(buffer::AbstractArray, offset::Int,
    acting_block_length::UInt16, acting_version::UInt16)
    GenericMessageDecoder(buffer, offset, Ref(0), acting_block_length, acting_version)
end
function GenericMessageDecoder(buffer::AbstractArray, offset::Int, hdr::MessageHeader)
    GenericMessageDecoder(buffer, offset, Ref(0), hdr)
end
GenericMessageDecoder(buffer::AbstractArray, hdr::MessageHeader) = GenericMessageDecoder(buffer, 0, Ref(0), hdr)
function GenericMessageEncoder(buffer::AbstractArray, position_ptr::Base.RefValue{Int64})
    GenericMessageEncoder(buffer, 0, position_ptr)
end
function GenericMessageEncoder(buffer::AbstractArray, offset::Int, position_ptr::Base.RefValue{Int64},
    hdr::MessageHeader)
    blockLength!(hdr, UInt16(0x40))
    templateId!(hdr, UInt16(0x7))
    schemaId!(hdr, UInt16(0x9))
    version!(hdr, UInt16(0x0))
    GenericMessageEncoder(buffer, offset + sbe_encoded_length(hdr), position_ptr)
end
function GenericMessageEncoder(buffer::AbstractArray, position_ptr::Base.RefValue{Int64}, hdr::MessageHeader)
    GenericMessageEncoder(buffer, 0, position_ptr, hdr)
end
function GenericMessageEncoder(buffer::AbstractArray, offset::Int, hdr::MessageHeader)
    GenericMessageEncoder(buffer, offset, Ref(0), hdr)
end
function GenericMessageEncoder(buffer::AbstractArray, hdr::MessageHeader)
    GenericMessageEncoder(buffer, 0, Ref(0), hdr)
end
GenericMessageEncoder(buffer::AbstractArray, offset::Int=0) = GenericMessageEncoder(buffer, offset, Ref(0))
sbe_buffer(m::GenericMessage) = m.buffer
sbe_offset(m::GenericMessage) = m.offset
sbe_position_ptr(m::GenericMessage) = m.position_ptr
sbe_position(m::GenericMessage) = m.position_ptr[]
sbe_position!(m::GenericMessage, position) = m.position_ptr[] = position
sbe_block_length(::GenericMessage) = UInt16(0x40)
sbe_block_length(::Type{<:GenericMessage}) = UInt16(0x40)
sbe_template_id(::GenericMessage) = UInt16(0x7)
sbe_template_id(::Type{<:GenericMessage})  = UInt16(0x7)
sbe_schema_id(::GenericMessage) = UInt16(0x9)
sbe_schema_id(::Type{<:GenericMessage})  = UInt16(0x9)
sbe_schema_version(::GenericMessage) = UInt16(0x0)
sbe_schema_version(::Type{<:GenericMessage})  = UInt16(0x0)
sbe_semantic_type(::GenericMessage) = ""
sbe_semantic_version(::GenericMessage) = ""
sbe_acting_block_length(m::GenericMessageDecoder) = m.acting_block_length
sbe_acting_block_length(::GenericMessageEncoder) = UInt16(0x40)
sbe_acting_version(m::GenericMessageDecoder) = m.acting_version
sbe_acting_version(::GenericMessageEncoder) = UInt16(0x0)
sbe_rewind!(m::GenericMessage) = sbe_position!(m, m.offset + m.acting_block_length)
sbe_rewind!(m::GenericMessageEncoder) = sbe_position!(m, m.offset + UInt16(0x40))
sbe_encoded_length(m::GenericMessage) = sbe_position(m) - m.offset
@inline function sbe_decoded_length(m::GenericMessage)
    skipper = GenericMessageDecoder(sbe_buffer(m), sbe_offset(m),
        sbe_acting_block_length(m), sbe_acting_version(m))
    sbe_rewind!(skipper)
    skip!(skipper)
    sbe_encoded_length(skipper)
end

function header_meta_attribute(::GenericMessage, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end
header_id(::GenericMessage) = UInt16(0x1)
header_since_version(::GenericMessage) = UInt16(0x0)
header_in_acting_version(m::GenericMessage) = sbe_acting_version(m) >= UInt16(0x0)
header_encoding_offset(::GenericMessage) = 0
header(m::GenericMessage) = SpidersMessageHeader(m.buffer, m.offset + 0, sbe_acting_version(m))

function show(io::IO, m::GenericMessage{T}) where {T}
    println(io, "GenericMessage view over a type $T")
    println(io, "SbeBlockLength: ", sbe_block_length(m))
    println(io, "SbeTemplateId:  ", sbe_template_id(m))
    println(io, "SbeSchemaId:    ", sbe_schema_id(m))
    println(io, "SbeSchemaVersion: ", sbe_schema_version(m))

    writer = GenericMessageDecoder(sbe_buffer(m), sbe_offset(m), sbe_block_length(m), sbe_schema_version(m))
    print(io, "header: ")
    show(io, header(writer))

    nothing
end

@inline function skip!(m::GenericMessage)
    return
end
