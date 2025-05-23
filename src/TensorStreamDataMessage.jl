# Generated SBE (Simple Binary Encoding) message codec
# Code generated by SBE. DO NOT EDIT.

export TensorStreamDataMessage, TensorStreamDataMessageDecoder, TensorStreamDataMessageEncoder
abstract type TensorStreamDataMessage{T} end

struct TensorStreamDataMessageDecoder{T<:AbstractArray{UInt8}} <: TensorStreamDataMessage{T}
    buffer::T
    offset::Int64
    position_ptr::Base.RefValue{Int64}
    acting_block_length::UInt16
    acting_version::UInt16
    function TensorStreamDataMessageDecoder(buffer::T, offset::Integer, position_ptr::Ref{Int64},
        acting_block_length::Integer, acting_version::Integer) where {T}
        position_ptr[] = offset + acting_block_length
        new{T}(buffer, offset, position_ptr, acting_block_length, acting_version)
    end
end

struct TensorStreamDataMessageEncoder{T<:AbstractArray{UInt8},HasSbeHeader} <: TensorStreamDataMessage{T}
    buffer::T
    offset::Int64
    position_ptr::Base.RefValue{Int64}
    function TensorStreamDataMessageEncoder(buffer::T, offset::Integer,
        position_ptr::Ref{Int64}, hasSbeHeader::Bool=false) where {T}
        position_ptr[] = offset + 80
        new{T,hasSbeHeader}(buffer, offset, position_ptr)
    end
end

@inline function TensorStreamDataMessageDecoder(buffer::AbstractArray, offset::Integer=0;
    position_ptr::Base.RefValue{Int64}=Ref(0),
    header::MessageHeader=MessageHeader(buffer, offset))
    if templateId(header) != UInt16(0xf) || schemaId(header) != UInt16(0x1)
        throw(DomainError("Template id or schema id mismatch"))
    end
    TensorStreamDataMessageDecoder(buffer, offset + sbe_encoded_length(header), position_ptr,
        blockLength(header), version(header))
end
@inline function TensorStreamDataMessageEncoder(buffer::AbstractArray, offset::Integer=0;
    position_ptr::Base.RefValue{Int64}=Ref(0),
    header::MessageHeader=MessageHeader(buffer, offset))
    blockLength!(header, UInt16(0x50))
    templateId!(header, UInt16(0xf))
    schemaId!(header, UInt16(0x1))
    version!(header, UInt16(0x0))
    TensorStreamDataMessageEncoder(buffer, offset + sbe_encoded_length(header), position_ptr, true)
end
sbe_buffer(m::TensorStreamDataMessage) = m.buffer
sbe_offset(m::TensorStreamDataMessage) = m.offset
sbe_position_ptr(m::TensorStreamDataMessage) = m.position_ptr
sbe_position(m::TensorStreamDataMessage) = m.position_ptr[]
sbe_position!(m::TensorStreamDataMessage, position) = m.position_ptr[] = position
sbe_block_length(::TensorStreamDataMessage) = UInt16(0x50)
sbe_block_length(::Type{<:TensorStreamDataMessage}) = UInt16(0x50)
sbe_template_id(::TensorStreamDataMessage) = UInt16(0xf)
sbe_template_id(::Type{<:TensorStreamDataMessage})  = UInt16(0xf)
sbe_schema_id(::TensorStreamDataMessage) = UInt16(0x1)
sbe_schema_id(::Type{<:TensorStreamDataMessage})  = UInt16(0x1)
sbe_schema_version(::TensorStreamDataMessage) = UInt16(0x0)
sbe_schema_version(::Type{<:TensorStreamDataMessage})  = UInt16(0x0)
sbe_semantic_type(::TensorStreamDataMessage) = ""
sbe_semantic_version(::TensorStreamDataMessage) = ""
sbe_acting_block_length(m::TensorStreamDataMessageDecoder) = m.acting_block_length
sbe_acting_block_length(::TensorStreamDataMessageEncoder) = UInt16(0x50)
sbe_acting_version(m::TensorStreamDataMessageDecoder) = m.acting_version
sbe_acting_version(::TensorStreamDataMessageEncoder) = UInt16(0x0)
sbe_rewind!(m::TensorStreamDataMessage) = sbe_position!(m, m.offset + sbe_acting_block_length(m))
sbe_encoded_length(m::TensorStreamDataMessage) = sbe_position(m) - m.offset
@inline function sbe_decoded_length(m::TensorStreamDataMessage)
    skipper = TensorStreamDataMessageDecoder(sbe_buffer(m), sbe_offset(m), Ref(0),
        sbe_acting_block_length(m), sbe_acting_version(m))
    sbe_skip!(skipper)
    sbe_encoded_length(skipper)
end

Base.sizeof(m::TensorStreamDataMessage) = sbe_decoded_length(m)
function Base.convert(::Type{AbstractArray{UInt8}}, m::TensorStreamDataMessageEncoder{<:AbstractArray{UInt8},true})
    return view(m.buffer, m.offset+1-sbe_encoded_length(MessageHeader):m.offset+sbe_encoded_length(m))
end
function Base.convert(::Type{AbstractArray{UInt8}}, m::TensorStreamDataMessageEncoder{<:AbstractArray{UInt8},false})
    return view(m.buffer, m.offset+1:m.offset+sbe_encoded_length(m))
end

function header_meta_attribute(::TensorStreamDataMessage, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end
header_id(::TensorStreamDataMessage) = UInt16(0x1)
header_since_version(::TensorStreamDataMessage) = UInt16(0x0)
header_in_acting_version(m::TensorStreamDataMessage) = sbe_acting_version(m) >= UInt16(0x0)
header_encoding_offset(::TensorStreamDataMessage) = 0
header(m::TensorStreamDataMessage) = SpidersMessageHeader(m.buffer, m.offset + 0, sbe_acting_version(m))

function sequencenumber_meta_attribute(::TensorStreamDataMessage, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end
sequencenumber_id(::TensorStreamDataMessage) = UInt16(0x2)
sequencenumber_since_version(::TensorStreamDataMessage) = UInt16(0x0)
sequencenumber_in_acting_version(m::TensorStreamDataMessage) = sbe_acting_version(m) >= UInt16(0x0)
sequencenumber_encoding_offset(::TensorStreamDataMessage) = 64
sequencenumber_null_value(::TensorStreamDataMessage) = Int64(-9223372036854775808)
sequencenumber_min_value(::TensorStreamDataMessage) = Int64(-9223372036854775807)
sequencenumber_max_value(::TensorStreamDataMessage) = Int64(9223372036854775807)
sequencenumber_encoding_length(::TensorStreamDataMessage) = 8

@inline function sequencenumber(m::TensorStreamDataMessageDecoder)
    return decode_le(Int64, m.buffer, m.offset + 64)
end
@inline sequencenumber!(m::TensorStreamDataMessageEncoder, value) = encode_le(Int64, m.buffer, m.offset + 64, value)

function offset_meta_attribute(::TensorStreamDataMessage, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end
offset_id(::TensorStreamDataMessage) = UInt16(0xb)
offset_since_version(::TensorStreamDataMessage) = UInt16(0x0)
offset_in_acting_version(m::TensorStreamDataMessage) = sbe_acting_version(m) >= UInt16(0x0)
offset_encoding_offset(::TensorStreamDataMessage) = 72
offset_null_value(::TensorStreamDataMessage) = UInt64(0xffffffffffffffff)
offset_min_value(::TensorStreamDataMessage) = UInt64(0x0)
offset_max_value(::TensorStreamDataMessage) = UInt64(0xfffffffffffffffe)
offset_encoding_length(::TensorStreamDataMessage) = 8

@inline function offset(m::TensorStreamDataMessageDecoder)
    return decode_le(UInt64, m.buffer, m.offset + 72)
end
@inline offset!(m::TensorStreamDataMessageEncoder, value) = encode_le(UInt64, m.buffer, m.offset + 72, value)

function chunk_meta_attribute(::TensorStreamDataMessage, meta_attribute)
    meta_attribute === :presence && return Symbol("required")
    return Symbol("")
end

chunk_character_encoding(::TensorStreamDataMessage) = "null"
chunk_in_acting_version(m::TensorStreamDataMessage) = sbe_acting_version(m) >= 0
chunk_id(::TensorStreamDataMessage) = 20
chunk_header_length(::TensorStreamDataMessage) = 4

@inline function chunk_length(m::TensorStreamDataMessage)
    return decode_le(UInt32, m.buffer, sbe_position(m))
end

@inline function chunk_length!(m::TensorStreamDataMessageEncoder, n)
    @boundscheck n > 1073741824 && throw(ArgumentError("length exceeds schema limit"))
    @boundscheck checkbounds(m.buffer, sbe_position(m) + 4 + n)
    return encode_le(UInt32, m.buffer, sbe_position(m), n)
end

@inline function skip_chunk!(m::TensorStreamDataMessageDecoder)
    len = chunk_length(m)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    return len
end

@inline function chunk(m::TensorStreamDataMessageDecoder)
    len = chunk_length(m)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    return view(m.buffer, pos+1:pos+len)
end

@inline chunk(m::TensorStreamDataMessageDecoder, ::Type{AbstractArray{T}}) where {T<:Real} = reinterpret(T, chunk(m))
@inline chunk(m::TensorStreamDataMessageDecoder, ::Type{NTuple{N,T}}) where {N,T<:Real} = (x = reinterpret(T, chunk(m)); ntuple(i -> x[i], Val(N)))
@inline chunk(m::TensorStreamDataMessageDecoder, ::Type{T}) where {T<:AbstractString} = StringView(rstrip_nul(chunk(m)))
@inline chunk(m::TensorStreamDataMessageDecoder, ::Type{T}) where {T<:Symbol} = Symbol(chunk(m, StringView))
@inline chunk(m::TensorStreamDataMessageDecoder, ::Type{T}) where {T<:Real} = reinterpret(T, chunk(m))[]
@inline chunk(m::TensorStreamDataMessageDecoder, ::Type{T}) where {T<:Nothing} = (skip_chunk!(m); nothing)

@inline function chunk_buffer!(m::TensorStreamDataMessageEncoder, len)
    chunk_length!(m, len)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    return view(m.buffer, pos+1:pos+len)
end

@inline function chunk!(m::TensorStreamDataMessageEncoder, src::AbstractArray)
    len = sizeof(src)
    chunk_length!(m, len)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    dest = view(m.buffer, pos+1:pos+len)
    copyto!(dest, reinterpret(UInt8, src))
end

@inline function chunk!(m::TensorStreamDataMessageEncoder, src::NTuple)
    len = sizeof(src)
    chunk_length!(m, len)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    dest = view(m.buffer, pos+1:pos+len)
    copyto!(dest, reinterpret(NTuple{len,UInt8}, src))
end

@inline function chunk!(m::TensorStreamDataMessageEncoder, src::AbstractString)
    len = sizeof(src)
    chunk_length!(m, len)
    pos = sbe_position(m) + 4
    sbe_position!(m, pos + len)
    dest = view(m.buffer, pos+1:pos+len)
    copyto!(dest, codeunits(src))
end

@inline chunk!(m::TensorStreamDataMessageEncoder, src::Symbol) = chunk!(m, to_string(src))
@inline chunk!(m::TensorStreamDataMessageEncoder, src::StaticString) = chunk!(m, Tuple(src))
@inline chunk!(m::TensorStreamDataMessageEncoder, src::Real) = chunk!(m, Tuple(src))
@inline chunk!(m::TensorStreamDataMessageEncoder, ::Nothing) = chunk_buffer!(m, 0)

function show(io::IO, m::TensorStreamDataMessage{T}) where {T}
    println(io, "TensorStreamDataMessage view over a type $T")
    println(io, "SbeBlockLength: ", sbe_block_length(m))
    println(io, "SbeTemplateId:  ", sbe_template_id(m))
    println(io, "SbeSchemaId:    ", sbe_schema_id(m))
    println(io, "SbeSchemaVersion: ", sbe_schema_version(m))

    writer = TensorStreamDataMessageDecoder(sbe_buffer(m), sbe_offset(m), Ref(0),
        sbe_block_length(m), sbe_schema_version(m))
    print(io, "header: ")
    show(io, header(writer))

    println(io)
    print(io, "sequencenumber: ")
    print(io, sequencenumber(writer))

    println(io)
    print(io, "offset: ")
    print(io, offset(writer))

    println(io)
    print(io, "chunk: ")
    print(io, skip_chunk!(writer))
    print(io, " bytes of raw data")

    nothing
end

@inline function sbe_skip!(m::TensorStreamDataMessageDecoder)
    sbe_rewind!(m)
    skip_chunk!(m)
    return
end
